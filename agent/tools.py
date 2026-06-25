"""
agent/tools.py — FunctionTool wrappers for the FunctionAgent.
Tools: execute_sql, get_schema, compute_stats.
"""
import json
import logging
import re
import time

from sqlalchemy import text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core tool functions
# ---------------------------------------------------------------------------

def execute_sql(sql: str, engine, row_cap: int = 200, tracker: dict = None) -> str:
    """Execute a SELECT query and return results as JSON."""
    start_time = time.time()
    logger.info("[TOOL:sql] Executing: %s...", sql[:120])

    try:
        stripped = sql.strip()

        if not stripped.upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT queries are allowed"})

        # Inject TOP N if not already present (SQL Server uses TOP, not LIMIT)
        if not re.search(r"\bTOP\s+\d+\b", stripped, re.IGNORECASE):
            stripped = re.sub(
                r"\bSELECT\b",
                f"SELECT TOP {row_cap}",
                stripped,
                count=1,
                flags=re.IGNORECASE,
            )

        # Remove LIMIT clause if agent accidentally used it (MySQL/Postgres habit)
        stripped = re.sub(r"\bLIMIT\s+\d+\b", "", stripped, flags=re.IGNORECASE).strip()

        with engine.connect() as conn:
            result = conn.execute(text(stripped))
            col_keys = list(result.keys())   # must read keys BEFORE fetchall
            rows = result.fetchall()

        result_data = [
            {col_keys[i]: row[i] for i in range(len(col_keys))}
            for row in rows
        ]

        # Update per-request tracker
        if tracker is not None:
            tracker["sql"] = stripped
            tracker["data"] = result_data
            matches = re.findall(
                r"\bFROM\s+\[?(\w+)\]?|\bJOIN\s+\[?(\w+)\]?",
                stripped,
                re.IGNORECASE,
            )
            for t1, t2 in matches:
                name = t1 or t2
                if name and name not in tracker["tables_used"]:
                    tracker["tables_used"].append(name)

        elapsed = (time.time() - start_time) * 1000
        logger.info("[TOOL:sql] %d rows in %.0fms", len(result_data), elapsed)

        return json.dumps(
            {"rows": result_data, "row_count": len(result_data), "columns": col_keys},
            default=str,
        )

    except Exception as e:
        error_msg = str(e)
        logger.warning("[TOOL:sql] Failed: %s", error_msg)

        # Helpful error for invalid table name
        match = re.search(r"Invalid object name '([^']+)'", error_msg)
        if match:
            return json.dumps({
                "error": f"Table '{match.group(1)}' not found in database",
                "suggestion": "Call get_schema('') to list all available tables, then retry with the correct name",
            })

        # Hint for LIMIT syntax error (SQL Server doesn't support LIMIT)
        if "LIMIT" in error_msg.upper() or "incorrect syntax" in error_msg.lower():
            return json.dumps({
                "error": f"SQL syntax error: {error_msg}",
                "suggestion": "Use TOP N instead of LIMIT. Example: SELECT TOP 100 * FROM Table WITH (NOLOCK)",
            })

        return json.dumps({
            "error": error_msg,
            "suggestion": "Check table/column names using get_schema. Use SQL Server syntax (TOP, NOLOCK, TRY_CAST, GETDATE)",
        })


def get_schema(table_name: str, schema_loader) -> str:
    """Get table schema details from YAML. Pass empty string to list all tables."""
    start_time = time.time()
    logger.info("[TOOL:schema] Getting: '%s'", table_name or "ALL_TABLES")

    try:
        contexts = schema_loader.get_all_table_contexts()

        if table_name and table_name.strip():
            name = table_name.strip()
            if name in contexts:
                result = {"table": name, "schema": contexts[name]}
            else:
                # Fuzzy match — find tables with similar names
                available = list(contexts.keys())
                similar = [t for t in available if name.lower() in t.lower()]
                result = {
                    "error": f"Table '{name}' not found",
                    "similar_tables": similar[:5],
                    "all_tables": available,
                }
        else:
            result = {
                "available_tables": list(contexts.keys()),
                "table_count": len(contexts),
                "usage": "Call get_schema('TableName') to see columns for a specific table",
            }

        elapsed = (time.time() - start_time) * 1000
        logger.info("[TOOL:schema] Completed in %.0fms", elapsed)
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("[TOOL:schema] Failed: %s", e)
        return json.dumps({"error": f"Schema lookup failed: {e}"})


def compute_stats(data_json: str, operation: str, column: str = None) -> str:
    """
    Calculate statistics on data returned by execute_sql.
    operation: sum | avg | average | min | max | count
    column: column name to operate on (auto-detects first numeric if omitted)
    """
    start_time = time.time()
    logger.info("[TOOL:stats] %s on column '%s'", operation, column or "auto")

    try:
        data = json.loads(data_json)
        rows = data.get("rows", []) if isinstance(data, dict) else data

        if not rows:
            return json.dumps({"error": "No data to compute statistics on", "result": None})

        # Auto-detect numeric column
        if not column:
            for key in rows[0].keys():
                try:
                    float(str(rows[0][key]).replace(",", ""))
                    column = key
                    break
                except (ValueError, TypeError):
                    continue

        if not column:
            return json.dumps({"error": "No numeric column found in data", "columns": list(rows[0].keys())})

        values = []
        for row in rows:
            try:
                v = str(row.get(column, "")).replace(",", "")
                values.append(float(v))
            except (ValueError, TypeError):
                continue

        if not values:
            return json.dumps({"error": f"No valid numeric values in column '{column}'"})

        op = operation.lower().strip()
        total = sum(values)
        avg = total / len(values)

        result = {
            "column": column,
            "operation": op,
            "count": len(values),
            "sum": round(total, 4),
            "average": round(avg, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

        if op in ("sum",):
            result["result"] = result["sum"]
        elif op in ("avg", "average", "mean"):
            result["result"] = result["average"]
        elif op == "min":
            result["result"] = result["min"]
        elif op == "max":
            result["result"] = result["max"]
        elif op == "count":
            result["result"] = result["count"]
        else:
            result["result"] = result["sum"]

        elapsed = (time.time() - start_time) * 1000
        logger.info("[TOOL:stats] count=%d, avg=%.2f in %.0fms", len(values), avg, elapsed)
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("[TOOL:stats] Failed: %s", e)
        return json.dumps({"error": f"Stats calculation failed: {e}"})


# ---------------------------------------------------------------------------
# FunctionTool factory functions
# ---------------------------------------------------------------------------

def create_sql_tool(engine, row_cap: int = 200, tracker: dict = None):
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=lambda sql: execute_sql(sql, engine, row_cap, tracker),
        name="execute_sql",
        description=(
            f"Execute a SELECT query against the Microsoft SQL Server database. "
            f"Returns rows as JSON (max {row_cap} rows). "
            "IMPORTANT SQL Server rules: use TOP N not LIMIT, add WITH (NOLOCK) after every table, "
            "use TRY_CAST(col AS FLOAT) for text-to-number conversion, use GETDATE() not NOW(). "
            "On error, check the suggestion field and retry with corrected SQL."
        ),
    )


def create_schema_tool(schema_loader):
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=lambda table_name: get_schema(table_name, schema_loader),
        name="get_schema",
        description=(
            "Get column names and descriptions for a database table. "
            "Pass an empty string '' to list ALL available tables. "
            "Pass a table name (e.g. 'ActivityTaskPlan') to see its columns. "
            "Always call this before writing SQL if you are unsure of column names."
        ),
    )


def create_stats_tool():
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=compute_stats,
        name="compute_stats",
        description=(
            "Calculate sum, average, min, max, or count on a JSON dataset from execute_sql. "
            "Parameters: data_json (the full JSON string from execute_sql), "
            "operation ('sum'|'avg'|'min'|'max'|'count'), "
            "column (column name — optional, auto-detects first numeric column if omitted)."
        ),
    )

import logging
import json
import re
import time
from sqlalchemy import text

logger = logging.getLogger(__name__)


def execute_sql(sql: str, engine, row_cap: int = 20, tracker: dict = None) -> str:
    """Execute a SELECT query and return results as JSON."""
    start_time = time.time()

    logger.info(f"[TOOL:sql] Executing: {sql[:100]}...")

    try:
        if not sql.strip().upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT queries are allowed"})

        if not re.search(r"\bTOP\s+\d+\b", sql, re.IGNORECASE):
            sql = re.sub(
                r"\bSELECT\b",
                f"SELECT TOP {row_cap}",
                sql,
                count=1,
                flags=re.IGNORECASE,
            )

        with engine.connect() as conn:
            result = conn.execute(text(sql))
            col_keys = list(result.keys())
            rows = result.fetchall()

        result_data = []
        for row in rows:
            row_dict = {}
            for i, key in enumerate(col_keys):
                row_dict[key] = row[i]
            result_data.append(row_dict)

        response = {
            "rows": result_data,
            "row_count": len(result_data),
            "columns": col_keys,
        }

        # Record the last successful SQL + data for the API response
        if tracker is not None:
            tracker["sql"] = sql
            tracker["data"] = result_data
            # Accumulate unique table names from SQL
            tables = re.findall(r"\bFROM\s+\[?(\w+)\]?|\bJOIN\s+\[?(\w+)\]?", sql, re.IGNORECASE)
            for t1, t2 in tables:
                name = t1 or t2
                if name and name not in tracker["tables_used"]:
                    tracker["tables_used"].append(name)

        logger.info(
            f"[TOOL:sql] Completed: {len(result_data)} rows in "
            f"{(time.time() - start_time) * 1000:.0f}ms"
        )
        return json.dumps(response, default=str)

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"[TOOL:sql] Failed: {error_msg}")

        match = re.search(r"Invalid object name '([^']+)'", error_msg)
        if match:
            table = match.group(1)
            return json.dumps(
                {
                    "error": f"Table '{table}' not found",
                    "suggestion": "Use get_schema to see available tables",
                }
            )

        return json.dumps(
            {"error": error_msg, "suggestion": "Check table/column names"}
        )


def get_schema(table_name: str, schema_loader) -> str:
    """Get table schema details from YAML."""
    start_time = time.time()

    logger.info(f"[TOOL:schema] Getting: {table_name or 'ALL_TABLES'}")

    try:
        contexts = schema_loader.get_all_table_contexts()

        if table_name:
            if table_name in contexts:
                result = {"table": table_name, "schema": contexts[table_name]}
            else:
                available = list(contexts.keys())[:10]
                return json.dumps(
                    {
                        "error": f"Table '{table_name}' not found",
                        "available_tables": available,
                    }
                )
        else:
            result = {"tables": list(contexts.keys()), "table_count": len(contexts)}

        logger.info(
            f"[TOOL:schema] Completed in {(time.time() - start_time) * 1000:.0f}ms"
        )
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"[TOOL:schema] Failed: {e}")
        return json.dumps({"error": f"Failed to get schema: {e}"})


def compute_stats(data_json: str, operation: str, column: str = None) -> str:
    """Calculate statistics on data."""
    start_time = time.time()

    logger.info(f"[TOOL:stats] Computing {operation} on {column or 'first numeric'}")

    try:
        data = json.loads(data_json)
        rows = data.get("rows", []) if isinstance(data, dict) else data

        if not rows:
            return json.dumps({"error": "No data to compute statistics on"})

        if not column:
            for key in rows[0].keys():
                try:
                    float(rows[0][key])
                    column = key
                    break
                except (ValueError, TypeError):
                    continue

        if not column:
            return json.dumps({"error": "No numeric column found"})

        values = []
        for row in rows:
            try:
                values.append(float(row.get(column, 0)))
            except (ValueError, TypeError):
                continue

        if not values:
            return json.dumps({"error": f"No valid numeric values in '{column}'"})

        result = {
            "column": column,
            "count": len(values),
            "sum": sum(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

        logger.info(
            f"[TOOL:stats] Completed: count={len(values)}, avg={result['average']:.2f} "
            f"in {(time.time() - start_time) * 1000:.0f}ms"
        )
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"[TOOL:stats] Failed: {e}")
        return json.dumps({"error": f"Stats calculation failed: {e}"})


def create_sql_tool(engine, row_cap: int = 20, tracker: dict = None):
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=lambda sql: execute_sql(sql, engine, row_cap, tracker),
        name="execute_sql",
        description=f"Execute a SELECT query against the database. Returns rows as JSON. Max {row_cap} rows returned.",
    )


def create_schema_tool(schema_loader):
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=lambda table_name: get_schema(table_name, schema_loader),
        name="get_schema",
        description="Get column names and descriptions for a table. Pass empty string to list all available tables.",
    )


def create_stats_tool():
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=compute_stats,
        name="compute_stats",
        description="Calculate sum, average, count, min, max on a JSON dataset returned by execute_sql.",
    )

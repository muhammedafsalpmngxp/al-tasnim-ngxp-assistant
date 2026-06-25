"""
agent/agent.py — FunctionAgent for complex multi-step queries.
Creates a fresh agent + isolated tracker per request (no shared state / race conditions).
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent system prompt — built dynamically from schema, no hardcoding
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an intelligent database assistant for AL TASNIM LLC.

You have access to a Microsoft SQL Server database (AppMasterDB_Local) containing
oil & gas well delivery operations data for Oman (Nimr/NMR, Marmul/MRM, Al Burj/ABJ clusters).

The database tracks: well schedules, crew assignments, activity progress, employees,
revenue, procurement, engineering approvals, and well monitoring reports.

## Your Tools
1. execute_sql  — Run SELECT queries against the database
2. get_schema   — Get column details for any table (pass '' to list all tables)
3. compute_stats — Calculate sum/average/min/max/count on fetched data

## SQL Server Syntax Rules — STRICTLY REQUIRED
WRONG → CORRECT:
- `LIMIT 20`              → `SELECT TOP 20` (put TOP after SELECT, not at end)
- `SELECT * FROM Table`   → `SELECT * FROM Table WITH (NOLOCK)` (every table needs NOLOCK)
- `SUM(manhours)`         → `SUM(TRY_CAST(manhours AS FLOAT))` (text columns need TRY_CAST)
- `NOW()`                 → `GETDATE()`
- `col1 || col2`          → `col1 + col2` (string concat uses +)
- `CAST(x AS INT)`        → `TRY_CAST(x AS INT)` (TRY_CAST returns NULL on fail, CAST throws error)
- `IFERROR(...)`          → `ISNULL(col, default)` for NULL handling

Full example of correct SQL:
SELECT TOP 200
    w.Well_ID,
    w.text AS well_name,
    TRY_CAST(w.progress AS FLOAT) * 100 AS progress_pct,
    w.target_end
FROM ActivityTaskPlan w WITH (NOLOCK)
WHERE w.type = 'W'
  AND w.actual_end = '1900-01-01'
ORDER BY w.target_end ASC

## Key Data Rules
- '1900-01-01' in actual_start or actual_end means NOT YET STARTED / NOT YET COMPLETED
- progress is stored as 0.0–1.0 (multiply by 100 for percentage)
- manhours columns may be stored as text — always use TRY_CAST
- Well IDs look like '33151' and appear in Well_ID, pdo_well_id, well code fields

## How to Answer Step by Step
1. If unsure of table/column names → call get_schema first
2. Write precise SQL with NOLOCK and correct TOP N
3. Execute and examine the data
4. If result is empty → think about why (wrong filter? wrong table?) and retry
5. If data has text numbers → use compute_stats or TRY_CAST in SQL
6. Form a clear, specific, data-backed answer

## Prediction and Analysis Approach
For "likely to finish late", "at risk", "predict completion":
- Step 1: Fetch tasks/wells with progress%, planned start, planned end, actual start
- Step 2: Calculate elapsed_pct = DATEDIFF(day, actual_start, GETDATE()) / DATEDIFF(day, actual_start, target_end) * 100
- Step 3: If elapsed_pct > progress_pct by more than 15 points → task is at risk of delay
- Step 4: Estimate completion = actual_start + (1 / weekly_velocity) weeks of remaining work

## Available Tables
{table_list}

## Important
- Never guess column names — use get_schema if uncertain
- If a query fails, read the error suggestion and fix the SQL
- If data is insufficient to answer, say so explicitly — never hallucinate
- Max rows per query: {row_cap} — add WHERE filters if you need to focus on specific records
- Always give a human-readable summary, not just raw data
"""


def _build_system_prompt(schema_loader, row_cap: int) -> str:
    try:
        names = schema_loader.get_table_names()
        table_list = "\n".join(f"  - {n}" for n in names)
    except Exception:
        table_list = "  (call get_schema('') to list all tables)"
    return _SYSTEM_PROMPT_TEMPLATE.format(table_list=table_list, row_cap=row_cap)


# ---------------------------------------------------------------------------
# Per-request agent execution — no shared state
# ---------------------------------------------------------------------------

async def ask_complex_agent(
    llm,
    schema_loader,
    engine,
    question: str,
    request_id: str,
    row_cap: int = 200,
) -> dict:
    """
    Create a fresh FunctionAgent + isolated tracker for this single request.
    Returns: {"response": str, "sql": str|None, "data": list, "tables_used": list}
    """
    from llama_index.core.agent import FunctionAgent
    from agent.tools import create_sql_tool, create_schema_tool, create_stats_tool

    start_time = time.time()
    logger.info("[AGENT:%s] Starting: %s", request_id, question[:80])

    # Isolated per-request tracker — zero shared state between requests
    tracker: dict = {"sql": None, "data": [], "tables_used": []}

    # Build tools bound to this request's tracker
    sql_tool = create_sql_tool(engine, row_cap, tracker)
    schema_tool = create_schema_tool(schema_loader)
    stats_tool = create_stats_tool()

    system_prompt = _build_system_prompt(schema_loader, row_cap)

    agent = FunctionAgent(
        tools=[sql_tool, schema_tool, stats_tool],
        llm=llm,
        verbose=True,
        timeout=120,
        system_prompt=system_prompt,
    )

    try:
        handler = agent.run(user_msg=question)
        response = await handler
        answer = str(response)

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "[AGENT:%s] Done in %.0fms | rows=%d | sql=%s",
            request_id,
            elapsed,
            len(tracker.get("data", [])),
            "yes" if tracker.get("sql") else "no",
        )

        return {
            "response": answer,
            "sql": tracker.get("sql"),
            "data": tracker.get("data", []),
            "tables_used": tracker.get("tables_used", []),
            "latency_ms": elapsed,
        }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error("[AGENT:%s] Failed after %.0fms: %s", request_id, elapsed, e)
        raise

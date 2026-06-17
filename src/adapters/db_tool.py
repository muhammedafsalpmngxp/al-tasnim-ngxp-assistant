"""Database tool — imports the db_assistant pipeline IN-PROCESS.

The orchestrator calls pipeline.ask() directly (no HTTP, no separate server).
The db_assistant reads its config from the single root .env via os.environ.
All SQL is SELECT-only; the LLM in the pipeline only parses intent, never writes SQL.
"""
import asyncio
import json
import logging
import os
import sys
import threading

from langchain_core.tools import tool

from ..config import settings
from .registry import register_tool

logger = logging.getLogger("orchestrator.tool.db")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_LOCK = threading.Lock()
_pipeline_ask = None
_pipeline_shutdown = None


def _load_pipeline() -> None:
    """Import the db_assistant pipeline once, thread-safely."""
    global _pipeline_ask, _pipeline_shutdown
    if _pipeline_ask is not None:
        return
    with _LOCK:
        if _pipeline_ask is not None:
            return

        # Ensure project root is on sys.path so tools.db_assistant resolves
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)

        # Load root .env into os.environ before the pipeline reads settings
        from dotenv import load_dotenv
        env_path = settings.DB_ENV_FILE
        if not os.path.isabs(env_path):
            env_path = os.path.join(_PROJECT_ROOT, env_path)
        if os.path.exists(env_path):
            load_dotenv(env_path, override=False)
            logger.debug("DB env loaded from %s", env_path)
        else:
            logger.warning("DB env file not found at %s — relying on process environment", env_path)

        from tools.db_assistant.src.pipeline import ask, shutdown
        _pipeline_ask = ask
        _pipeline_shutdown = shutdown
        logger.info("db_assistant pipeline loaded successfully")


@register_tool
@tool
async def query_database(question: str) -> str:
    """Query the LIVE operational database for current well-delivery data:
    well status, rig assignments, depth/progress, counts, equipment, crews and
    latest readings.

    Use for questions like:
    - "What is the status of rig 104?"
    - "How many wells are currently active?"
    - "Show me all employees in Muscat"
    - "What equipment is available?"
    - "Current progress of Well 30750"

    Do NOT use for procedures, how-to questions, or definitions — use search_documents.

    Args:
        question: A clear natural-language data question for the database.
    """
    logger.info("[DB_TOOL] CALL question=%r", question)

    try:
        await asyncio.to_thread(_load_pipeline)
        result = await asyncio.wait_for(
            asyncio.to_thread(_pipeline_ask, question),
            timeout=settings.DB_TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        msg = f"Database query timed out after {settings.DB_TOOL_TIMEOUT}s"
        logger.error("[DB_TOOL] %s", msg)
        return json.dumps({"status": "error", "error": msg})
    except Exception as e:
        logger.error("[DB_TOOL] pipeline error: %s", e, exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})

    if getattr(result, "clarification_needed", False):
        logger.info("[DB_TOOL] needs clarification: %s", result.clarification_question)
        return json.dumps({
            "status": "needs_clarification",
            "question": result.clarification_question,
            "suggestions": result.suggestions or [],
        })

    if not result.success:
        logger.warning("[DB_TOOL] error: %s", result.error)
        return json.dumps({"status": "error", "error": result.error or "Unknown error"})

    logger.info(
        "[DB_TOOL] ok rows=%s sql=%r",
        result.row_count, (result.sql or "")[:120],
    )
    return json.dumps({
        "status": "ok",
        "answer": result.answer,
        "sql": result.sql,
        "row_count": result.row_count,
        "rows": result.data,
        "source": "operational_database",
    })


def close_pipeline() -> None:
    """Release DB connections. Called from orchestrator shutdown."""
    if _pipeline_shutdown:
        _pipeline_shutdown()

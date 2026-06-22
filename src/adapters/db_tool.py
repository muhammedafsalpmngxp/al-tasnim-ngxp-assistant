"""Database tool — imports the db_assistant pipeline IN-PROCESS.

The orchestrator calls pipeline.ask() directly (no HTTP, no separate server).
The db_assistant reads its config from the single root .env via os.environ.
All SQL is SELECT-only; the LLM in the pipeline only parses intent, never writes SQL.
"""
import asyncio
import decimal
import json
import logging
import os
import sys
import threading
from datetime import date, datetime, time
from typing import Optional

from langchain_core.tools import tool

from ..config import settings
from .registry import register_tool

logger = logging.getLogger("orchestrator.tool.db")


def _json_default(obj):
    """JSON encoder for SQL result types not handled by the stdlib encoder."""
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_LOCK = threading.Lock()
_pipeline_ask = None
_pipeline_shutdown = None

# Event set when init completes (success or failure), so query_database can wait.
_init_done = threading.Event()
_init_error: Optional[Exception] = None


def _load_pipeline() -> None:
    """Import + fully initialize the db_assistant pipeline once, thread-safely."""
    global _pipeline_ask, _pipeline_shutdown, _init_error

    if _pipeline_ask is not None:
        return

    with _LOCK:
        if _pipeline_ask is not None:
            return

        try:
            if _PROJECT_ROOT not in sys.path:
                sys.path.insert(0, _PROJECT_ROOT)

            from dotenv import load_dotenv
            env_path = settings.DB_ENV_FILE
            if not os.path.isabs(env_path):
                env_path = os.path.join(_PROJECT_ROOT, env_path)
            if os.path.exists(env_path):
                load_dotenv(env_path, override=True)
                logger.debug("DB env loaded from %s", env_path)
            else:
                logger.warning("DB env file not found at %s — relying on process env", env_path)

            # Import triggers module-level _load_env() which walks up to find .env
            from tools.db_assistant.src.pipeline import ask, shutdown, _initialize

            # Run full initialization now (schema discovery, index build, etc.)
            _initialize()

            _pipeline_ask = ask
            _pipeline_shutdown = shutdown
            logger.info("db_assistant pipeline ready")

        except Exception as exc:
            _init_error = exc
            logger.error("Pipeline initialization failed: %s", exc, exc_info=True)
            raise
        finally:
            _init_done.set()  # always unblock waiters, even on failure


def start_pipeline_init() -> None:
    """Launch pipeline initialization in a background thread at server startup.

    This decouples the (potentially long) first-run init from the first user
    request, so DB_TOOL_TIMEOUT only applies to the ask() call itself.
    """
    if _pipeline_ask is not None or _init_done.is_set():
        return
    t = threading.Thread(target=_load_pipeline, name="pipeline-init", daemon=True)
    t.start()
    logger.info("Pipeline initialization started in background (thread=%s)", t.name)


# Max seconds to wait for the pipeline to finish initializing before a query.
# Must be large enough for first-run: model download + schema discovery + index build.
_INIT_WAIT_SEC = int(os.getenv("PIPELINE_INIT_WAIT_SEC", "1800"))  # 30 min default


@register_tool
@tool
async def query_database(question: str) -> str:
    """Query live operational database records: well IDs, rig assignments, status, progress, activity codes, station codes, field names, crew details, dates, counts, and any structured operational data."""
    logger.info("[DB_TOOL] CALL question=%r", question)

    # If still initializing, wait patiently (separate from DB_TOOL_TIMEOUT)
    if not _init_done.is_set():
        logger.info("[DB_TOOL] Pipeline still initializing — waiting up to %ss ...", _INIT_WAIT_SEC)
        ready = await asyncio.to_thread(lambda: _init_done.wait(timeout=_INIT_WAIT_SEC))
        if not ready:
            msg = (
                f"Pipeline is still starting up after {_INIT_WAIT_SEC}s. "
                "This happens on the very first run while indexes are being built. "
                "Please try again in a moment."
            )
            logger.error("[DB_TOOL] %s", msg)
            return json.dumps({"status": "error", "error": msg}, default=_json_default)

    if _init_error is not None:
        msg = f"Pipeline initialization failed: {_init_error}"
        logger.error("[DB_TOOL] %s", msg)
        return json.dumps({"status": "error", "error": msg}, default=_json_default)

    # Pipeline ready — run the actual query with DB_TOOL_TIMEOUT
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_pipeline_ask, question),
            timeout=settings.DB_TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        msg = f"Database query timed out after {settings.DB_TOOL_TIMEOUT}s"
        logger.error("[DB_TOOL] %s", msg)
        return json.dumps({"status": "error", "error": msg}, default=_json_default)
    except Exception as e:
        logger.error("[DB_TOOL] pipeline error: %s", e, exc_info=True)
        return json.dumps({"status": "error", "error": str(e)}, default=_json_default)

    if getattr(result, "clarification_needed", False):
        logger.info("[DB_TOOL] needs clarification: %s", result.clarification_question)
        return json.dumps({
            "status": "needs_clarification",
            "question": result.clarification_question,
            "suggestions": result.suggestions or [],
        }, default=_json_default)

    if not result.success:
        logger.warning("[DB_TOOL] error: %s", result.error)
        return json.dumps({"status": "error", "error": result.error or "Unknown error"}, default=_json_default)

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
        "tables_used": result.tables_used or [],
        "source": "operational_database",
    }, default=_json_default)


def get_table_retriever():
    """Return the pipeline's LlamaIndex table retriever for semantic routing.

    Returns None if the pipeline has not finished initialising yet.
    The retriever is the same vector index used internally for SQL table selection,
    so routing and SQL generation share a single consistent embedding space.
    """
    if not _init_done.is_set() or _init_error is not None or _pipeline_ask is None:
        return None
    try:
        from tools.db_assistant.src.pipeline import _raw_table_retriever
        return _raw_table_retriever
    except Exception:
        return None


def close_pipeline() -> None:
    """Release DB connections. Called from orchestrator shutdown."""
    if _pipeline_shutdown:
        _pipeline_shutdown()

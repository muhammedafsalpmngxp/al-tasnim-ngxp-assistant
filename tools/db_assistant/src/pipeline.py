"""
Pure pipeline — no FastAPI dependency.

Imported in-process by the LangGraph orchestrator's query_database tool.
Can also be used directly in scripts/tests.
"""
import logging
import time
from typing import Optional

from .config import settings
from .connection_pool import ConnectionPool
from .conversation import ConversationManager
from .db_executor import ReadOnlyExecutor
from .intent_parser import IntentParser
from .models import FinalResponse
from .response_formatter import ResponseFormatter
from .schema_introspector import SchemaIntrospector
from .sql_builder import SQLBuilder
from .value_resolver import ValueResolver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline components — initialised ONCE at first import
# ---------------------------------------------------------------------------

_conn_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={settings.DB_SERVER};"
    f"DATABASE={settings.DB_NAME};"
    f"UID={settings.DB_READONLY_USER};"
    f"PWD={settings.DB_READONLY_PASSWORD};"
    f"ApplicationIntent=ReadOnly;"
)

pool = ConnectionPool(_conn_string, max_size=settings.POOL_SIZE)
schema = SchemaIntrospector(pool, settings.ALLOWED_TABLES)
resolver = ValueResolver(pool, settings.FUZZY_THRESHOLD, settings.AMBIGUITY_MARGIN)
conversation = ConversationManager()
parser = IntentParser(resolver, schema)
builder = SQLBuilder(schema, settings.MAX_ROWS, settings.PII_PATTERNS)
executor = ReadOnlyExecutor(pool, settings.QUERY_TIMEOUT_SEC)
formatter = ResponseFormatter()

logger.info(
    "DB pipeline ready (server=%s db=%s provider=%s)",
    settings.DB_SERVER, settings.DB_NAME, settings.LLM_PROVIDER,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str, session_id: Optional[str] = None) -> FinalResponse:
    """Process a natural-language question and return a FinalResponse.

    This is the single entry-point called by the orchestrator tool and by the
    standalone /ask HTTP endpoint.
    """
    start = time.time()
    sid = conversation.get_or_create_session(session_id)
    conversation.add_query(sid, question)

    clarification_context = None
    if conversation.needs_clarification(sid):
        clarification_context = conversation.get_clarification_context(sid)
        if not conversation.can_continue(sid):
            conversation.clear_clarification(sid)
            return FinalResponse(
                success=False,
                error="Too many clarification attempts. Please start with a complete question.",
                execution_time_ms=(time.time() - start) * 1000,
            )

    logger.debug("[DB] question=%r session=%s", question, sid)
    intent_response = parser.parse(question, clarification_context)
    logger.debug(
        "[DB] intent success=%s clarification=%s error=%r",
        intent_response.success, intent_response.clarification_needed, intent_response.error,
    )

    if intent_response.clarification_needed:
        conversation.set_clarification(
            sid,
            intent_response.clarification_question,
            question,
            intent_response.intent,
        )
        return FinalResponse(
            success=False,
            clarification_needed=True,
            clarification_question=intent_response.clarification_question,
            suggestions=intent_response.suggestions,
            execution_time_ms=(time.time() - start) * 1000,
        )

    if not intent_response.success:
        return FinalResponse(
            success=False,
            error=intent_response.error,
            execution_time_ms=(time.time() - start) * 1000,
        )

    conversation.clear_clarification(sid)

    try:
        sql, params = builder.build(intent_response.intent)
        logger.debug("[DB] SQL built: %s | params: %s", sql.strip(), params)
    except Exception as e:
        logger.error("[DB] SQL build failed: %s", e)
        return FinalResponse(
            success=False,
            error=f"Query build failed: {e}",
            execution_time_ms=(time.time() - start) * 1000,
        )

    result = executor.execute(sql, params)
    logger.debug(
        "[DB] execution success=%s rows=%s error=%r",
        result["success"], result.get("row_count"), result.get("error"),
    )

    if not result["success"]:
        return FinalResponse(
            success=False,
            error=result["error"],
            sql=sql,
            execution_time_ms=(time.time() - start) * 1000,
        )

    answer = formatter.format(
        question,
        result["data"],
        result["columns"],
        result["row_count"],
        intent_response.intent.table,
    )
    logger.debug("[DB] answer ready (%.0f ms)", (time.time() - start) * 1000)

    return FinalResponse(
        success=True,
        answer=answer,
        sql=sql,
        data=result["data"][:10],
        row_count=result["row_count"],
        execution_time_ms=(time.time() - start) * 1000,
    )


def shutdown() -> None:
    """Release DB connections. Call on application shutdown."""
    pool.close_all()
    logger.info("DB pipeline shut down")

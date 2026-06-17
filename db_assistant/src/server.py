from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time
import logging

from .config import settings
from .connection_pool import ConnectionPool
from .schema_introspector import SchemaIntrospector
from .value_resolver import ValueResolver
from .conversation import ConversationManager
from .intent_parser import IntentParser
from .sql_builder import SQLBuilder
from .db_executor import ReadOnlyExecutor
from .response_formatter import ResponseFormatter
from .models import FinalResponse

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

conn_string = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={settings.DB_SERVER};"
    f"DATABASE={settings.DB_NAME};"
    f"UID={settings.DB_READONLY_USER};"
    f"PWD={settings.DB_READONLY_PASSWORD};"
    f"ApplicationIntent=ReadOnly;"
)

pool = ConnectionPool(conn_string, max_size=settings.POOL_SIZE)
schema = SchemaIntrospector(pool, settings.ALLOWED_TABLES)
resolver = ValueResolver(pool, settings.FUZZY_THRESHOLD, settings.AMBIGUITY_MARGIN)
conversation = ConversationManager()
parser = IntentParser(resolver, schema)
builder = SQLBuilder(schema, settings.MAX_ROWS, settings.PII_PATTERNS)
executor = ReadOnlyExecutor(pool, settings.QUERY_TIMEOUT_SEC)
formatter = ResponseFormatter()

app = FastAPI(title="Database Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"])

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"

@app.post("/ask")
def ask(request: QueryRequest):
    start = time.time()
    session_id = conversation.get_or_create_session(request.session_id)
    conversation.add_query(session_id, request.question)
    
    clarification_context = None
    if conversation.needs_clarification(session_id):
        clarification_context = conversation.get_clarification_context(session_id)
        if not conversation.can_continue(session_id):
            conversation.clear_clarification(session_id)
            return FinalResponse(
                success=False,
                error="I've asked multiple times. Please start over with a complete question.",
                execution_time_ms=(time.time() - start) * 1000
            )
    
    logger.debug("[PIPELINE] question=%r session=%s", request.question, session_id)

    intent_response = parser.parse(request.question, clarification_context)
    logger.debug("[PIPELINE] intent_response: success=%s clarification=%s error=%r",
                 intent_response.success, intent_response.clarification_needed, intent_response.error)

    if intent_response.clarification_needed:
        conversation.set_clarification(
            session_id,
            intent_response.clarification_question,
            request.question,
            intent_response.intent
        )
        return FinalResponse(
            success=False,
            clarification_needed=True,
            clarification_question=intent_response.clarification_question,
            suggestions=intent_response.suggestions,
            execution_time_ms=(time.time() - start) * 1000
        )

    if not intent_response.success:
        return FinalResponse(
            success=False,
            error=intent_response.error,
            execution_time_ms=(time.time() - start) * 1000
        )

    conversation.clear_clarification(session_id)

    try:
        sql, params = builder.build(intent_response.intent)
        logger.debug("[PIPELINE] SQL built: %s | params: %s", sql, params)
    except Exception as e:
        logger.error("[PIPELINE] SQL build failed: %s", e)
        return FinalResponse(
            success=False,
            error=f"Failed to build query: {str(e)}",
            execution_time_ms=(time.time() - start) * 1000
        )

    result = executor.execute(sql, params)
    logger.debug("[PIPELINE] Execution result: success=%s row_count=%s error=%r",
                 result['success'], result.get('row_count'), result.get('error'))
    if not result['success']:
        return FinalResponse(
            success=False,
            error=result['error'],
            sql=sql,
            execution_time_ms=(time.time() - start) * 1000
        )

    answer = formatter.format(
        request.question,
        result['data'],
        result['columns'],
        result['row_count'],
        intent_response.intent.table
    )
    logger.debug("[PIPELINE] Formatted answer: %r", answer[:200] if answer else None)

    return FinalResponse(
        success=True,
        answer=answer,
        sql=sql,
        data=result['data'][:10],
        row_count=result['row_count'],
        execution_time_ms=(time.time() - start) * 1000
    )

@app.get("/health")
def health():
    from .llm_client import get_llm_client
    llm = get_llm_client()
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_available": llm.check_available(),
        "pool": pool.get_stats(),
    }

@app.post("/reset/{session_id}")
def reset(session_id: str):
    conversation._sessions.pop(session_id, None)
    return {"status": "reset"}

@app.on_event("shutdown")
def shutdown():
    pool.close_all()

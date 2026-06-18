"""
AL TASNIM Agentic AI — single FastAPI application.

Endpoints
---------
POST /chat    Natural-language query → LangGraph agent → answer
GET  /health  Liveness + dependency check
GET  /tools   List registered tools
"""
import logging
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage

from .adapters import tool_names
from .adapters.db_tool import close_pipeline
from .config import settings
from .evidence import collect_evidence, extract_final_answer, extract_tools_used, validate_answer
from .graph import build_app
from .observability import configure_logging, configure_tracing, correlation_id_var
from .schemas import ChatRequest, ChatResponse

configure_logging()
configure_tracing()
logger = logging.getLogger("orchestrator.api")

# ---------------------------------------------------------------------------
# App & graph — compiled once at startup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AL TASNIM Agentic AI",
    version="2.0.0",
    description="LangGraph orchestrator with live DB and document tools.",
)

_graph = build_app()
logger.info(
    "Orchestrator ready — provider=%s model=%s tools=%s",
    settings.LLM_PROVIDER, settings.LLM_MODEL, tool_names(),
)


# ---------------------------------------------------------------------------
# Middleware — correlation ID propagation
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _add_correlation_id(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    token = correlation_id_var.set(cid)
    try:
        response = await call_next(request)
    finally:
        correlation_id_var.reset(token)
    response.headers["X-Correlation-ID"] = cid
    return response


# ---------------------------------------------------------------------------
# /chat  — main endpoint
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a question; the agent decides which tools to call and returns an answer."""
    start = time.time()
    session_id = req.session_id or "default"
    logger.info("[CHAT] question=%r session=%s user=%s", req.question, session_id, req.user_id)

    try:
        state = {
            "messages": [HumanMessage(content=req.question)],
            "iterations": 0,
        }
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": settings.RECURSION_LIMIT,
        }

        logger.debug("[CHAT] invoking graph (session=%s)", session_id)
        result = await _graph.ainvoke(state, config)

        messages = result["messages"]
        evidence = collect_evidence(messages)
        answer = extract_final_answer(messages)
        tools_used = extract_tools_used(evidence)
        grounded, final_answer = validate_answer(answer, evidence)
        took = (time.time() - start) * 1000

        logger.info(
            "[CHAT] done iterations=%s tools=%s grounded=%s took=%.0fms",
            result.get("iterations", 0),
            [t.tool for t in tools_used],
            grounded,
            took,
        )

        return ChatResponse(
            success=True,
            answer=final_answer,
            tools_used=tools_used,
            iterations=result.get("iterations", 0),
            session_id=session_id,
            execution_time_ms=took,
        )

    except Exception as e:
        logger.exception("[CHAT] failed")
        return ChatResponse(
            success=False,
            error=str(e),
            session_id=session_id,
            execution_time_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    rag_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{settings.RAG_TOOL_URL}/health")
            rag_ok = r.status_code == 200
    except Exception:
        pass

    active_model = settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER == "gemini" else settings.LLM_MODEL
    body = {
        "status": "healthy",
        "provider": settings.LLM_PROVIDER,
        "model": active_model,
        "tools": tool_names(),
        "rag_reachable": rag_ok,
    }
    logger.debug("[HEALTH] %s", body)
    return body


# ---------------------------------------------------------------------------
# /tools
# ---------------------------------------------------------------------------

@app.get("/tools")
async def tools():
    return {"tools": tool_names()}


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def on_shutdown():
    close_pipeline()
    logger.info("Orchestrator shut down")

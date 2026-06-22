"""
AL TASNIM Agentic AI — FastAPI entry point.

Start:
    conda activate v12
    python agent/app.py

Or with uvicorn directly:
    uvicorn agent.app:app --port 8001 --reload

Endpoints:
    POST /agent/ask       — main agentic query endpoint
    GET  /agent/health    — health check
    GET  /agent/config    — show active config (admin only)
    GET  /docs            — Swagger UI
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Allow running as `python agent/app.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import get_config, server_cfg
from agent.models import AgentRequest, AgentResponse, UserRole
from agent.orchestrator import run_agent

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("agent.app")

# ── FastAPI app ───────────────────────────────────────────────────────────────
_cfg = server_cfg()

app = FastAPI(
    title=_cfg.get("title", "AL TASNIM Agentic AI"),
    version=_cfg.get("version", "1.0.0"),
    description=(
        "Controlled Agentic AI Operational Intelligence Assistant for "
        "well delivery, drilling progress, planning, and decision support. "
        "Implements the AL TASNIM Real-Time Agentic AI Architecture v2.1."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/agent/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "AL TASNIM Agentic AI", "version": _cfg.get("version", "1.0.0")}


@app.get("/agent/config", tags=["System"])
async def show_config(role: str = "operations"):
    """
    Show active agent configuration.
    Admin role sees full config; other roles see a safe summary.
    """
    if role == UserRole.ADMIN.value:
        cfg = get_config()
        # Redact LLM credentials
        safe = dict(cfg)
        if "llm" in safe:
            safe["llm"] = {k: {**v, "base_url": "***"} for k, v in cfg["llm"].items()}
        return safe
    return {
        "service":  _cfg.get("title"),
        "version":  _cfg.get("version"),
        "rag_base": get_config()["rag_service"]["base_url"],
        "cache":    get_config()["cache"],
    }


@app.post(
    "/agent/ask",
    response_model=AgentResponse,
    tags=["Agent"],
    summary="Ask the AL TASNIM Agentic AI",
    description="""
Submit a natural language query about AL TASNIM well delivery operations.

The agent will:
1. Check if clarification is needed (vague queries are rejected with a follow-up question)
2. Route to the correct tool(s): document RAG, SQL database, analytics, or recommendation
3. Validate evidence before answering
4. Return a structured response with confidence level, assumptions, and recommended action
5. Flag high-risk decisions for human review

**Query examples:**
- `"How many wells are currently in progress on RIG-101?"` → SQL tool
- `"What is the procedure for FLAF approval?"` → Document RAG tool
- `"When will the Nimr cluster activity finish? Well ID 31722"` → SQL + Analytics
- `"Why is well 30750 behind schedule?"` → SQL + Documents (Multi)
- `"What should we do to recover the delay on cluster ALBRG?"` → Recommend route
    """,
)
async def ask_agent(req: AgentRequest) -> AgentResponse:
    logger.info(
        "AGENT ask | session=%s role=%s | %s",
        req.session_id, req.user_role.value, req.query[:120],
    )

    result = await run_agent(
        query      = req.query,
        session_id = req.session_id,
        user_role  = req.user_role.value,
        top_k      = req.top_k,
    )

    return AgentResponse(
        query                  = req.query,
        direct_answer          = result.get("direct_answer", ""),
        evidence               = result.get("evidence", []),
        sources                = result.get("sources", []),
        confidence             = result.get("confidence", "Medium"),
        assumptions            = result.get("assumptions", []),
        risk_limitation        = result.get("risk_limitation") or None,
        recommended_next_action = result.get("next_action") or None,
        route                  = result.get("route", ""),
        tools_called           = result.get("tools_called", []),
        needs_human_review     = result.get("needs_human_review", False),
        human_review_reason    = result.get("human_review_reason") or None,
        from_cache             = result.get("from_cache", False),
        session_id             = req.session_id,
        clarification_needed   = result.get("clarification_needed", False),
        clarification_text     = result.get("clarification_text") or None,
        error                  = result.get("error") or None,
        latency_ms             = result.get("latency_ms", 0),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "agent.app:app",
        host=_cfg.get("host", "0.0.0.0"),
        port=_cfg.get("port", 8001),
        reload=True,
        log_level="info",
    )

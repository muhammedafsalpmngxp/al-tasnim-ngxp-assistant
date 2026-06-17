"""Standalone FastAPI wrapper for the db_assistant pipeline.

Used ONLY when running the db_assistant as an independent service (port 8000).
When the orchestrator is running, use pipeline.py directly — do not start this.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .config import settings
from .pipeline import ask, pool, shutdown

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(title="Database Assistant (standalone)", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"])


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"


@app.post("/ask")
def ask_endpoint(request: QueryRequest):
    return ask(request.question, request.session_id)


@app.get("/health")
def health():
    try:
        from .llm_client import get_llm_client
        llm_available = get_llm_client().check_available()
    except Exception as e:
        logger.warning("Health check: LLM unavailable — %s", e)
        llm_available = False
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_available": llm_available,
        "pool": pool.get_stats(),
    }


@app.on_event("shutdown")
def on_shutdown():
    shutdown()

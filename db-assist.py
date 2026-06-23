"""
db-assist.py — FastAPI application entry point for the AL TASNIM NL2SQL assistant.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from pipeline import NL2SQLPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_pipeline: NL2SQLPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the NL2SQL pipeline on startup; clean up on shutdown."""
    global _pipeline
    logger.info("Starting AL TASNIM DB Assistant ...")
    _pipeline = NL2SQLPipeline(settings)
    logger.info("Pipeline ready. Serving on %s:%d", settings.APP_HOST, settings.APP_PORT)
    yield
    logger.info("Shutting down AL TASNIM DB Assistant.")
    _pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AL TASNIM DB Assistant",
    description="Natural-language interface for the AppMasterDB_Local SQL Server database.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")


class AskResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    data: Optional[list[dict[str, Any]]] = None
    tables_used: Optional[list[str]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health() -> dict[str, Any]:
    """Liveness/readiness probe."""
    if _pipeline is None or not _pipeline.ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        tables_loaded = len(_pipeline._schema_loader.get_table_names())
    except Exception:
        tables_loaded = -1

    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "db_name": settings.DB_NAME,
        "tables_loaded": tables_loaded,
    }


@app.post("/ask", response_model=AskResponse, tags=["query"])
async def ask(request: AskRequest) -> AskResponse:
    """
    Submit a natural-language question and receive a structured response.

    The assistant will:
    1. Classify the intent (greeting / SQL / unrelated / ...)
    2. Retrieve relevant tables via hybrid BM25 + dense retrieval
    3. Generate and validate a SELECT query
    4. Execute it against AppMasterDB_Local
    5. Return a human-readable answer, the SQL used, and the raw data rows
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")

    try:
        result = await _pipeline.ask(request.question)
        return AskResponse(**result)
    except Exception as exc:
        logger.exception("Unhandled error processing question: %s", request.question)
        return AskResponse(
            answer="An unexpected error occurred. Please try again.",
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s", request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "db-assist:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )

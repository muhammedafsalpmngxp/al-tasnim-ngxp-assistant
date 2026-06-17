"""RAG document tool — SKELETON. Implements the stable contract the orchestrator expects.

Replace `search()` with your hybrid retrieval (vector + BM25 + optional rerank).
Until then it returns a well-formed empty result so the orchestrator runs end-to-end.

Contract:
    POST /search { "query": str, "top_k": int }
        -> { "success": true, "passages": [{"text","source","score"}], "sources": [str] }
    GET  /health -> { "status": "healthy" }
"""
import logging
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="AL TASNIM RAG Document Tool", version="0.1.0")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = settings.TOP_K


class Passage(BaseModel):
    text: str
    source: str
    score: float


class SearchResponse(BaseModel):
    success: bool = True
    passages: List[Passage] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None


def search(query: str, top_k: int) -> SearchResponse:
    # TODO (your team): hybrid retrieval over SOPs/KT docs goes here.
    logger.info("RAG skeleton search: %r (top_k=%s)", query, top_k)
    return SearchResponse(success=True, passages=[], sources=[])


@app.post("/search", response_model=SearchResponse)
async def do_search(req: SearchRequest):
    try:
        return search(req.query, req.top_k)
    except Exception as e:
        logger.exception("search failed")
        return SearchResponse(success=False, error=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "rag_assistant"}

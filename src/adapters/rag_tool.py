"""Document RAG tool — calls the rag_assistant HTTP service (stable contract)."""
import json
import logging

import httpx
from langchain_core.tools import tool

from ..config import settings
from ._resilience import with_timeout
from .registry import register_tool

logger = logging.getLogger("orchestrator.tool.rag")


@register_tool
@tool
async def search_documents(query: str) -> str:
    """Search engineering documents, SOPs, procedures, KT transcripts and
    training material. Use for how-to questions, procedures, explanations and
    definitions (e.g. "casing running procedure", "define spud date").
    Do NOT use for live operational status — use query_database for that.

    Args:
        query: The search query describing the procedure/topic to look up.
    """
    logger.info("CALL query=%r", query)

    async def _call():
        async with httpx.AsyncClient(timeout=settings.RAG_TOOL_TIMEOUT) as client:
            resp = await client.post(
                f"{settings.RAG_TOOL_URL}/search",
                json={"query": query, "top_k": settings.RAG_TOP_K},
            )
            resp.raise_for_status()
            return resp.json()

    try:
        d = await with_timeout(
            _call,
            timeout=settings.RAG_TOOL_TIMEOUT,
            retries=settings.RAG_RETRIES,
            backoff=1.0,
            retry_on=(httpx.HTTPError,),
        )
    except Exception as e:
        logger.error("search_documents failed: %s", e)
        return json.dumps({"status": "error", "error": str(e)})

    if not d.get("success", True):
        logger.warning("RESULT error: %s", d.get("error"))
        return json.dumps({"status": "error", "error": d.get("error")})

    logger.info("RESULT ok passages=%d", len(d.get("passages", [])))
    return json.dumps({
        "status": "ok",
        "passages": d.get("passages", []),
        "sources": d.get("sources", []),
        "source": "documents",
    })

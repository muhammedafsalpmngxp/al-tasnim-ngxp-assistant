"""Semantic routing node.

Uses the pipeline's raw vector table index (same embeddings used for SQL table selection)
to decide which tool the agent should call — before any LLM call is made.

The retriever used here (_raw_table_retriever) is a VectorIndexRetriever that returns
NodeWithScore objects with cosine similarity scores. This is different from the ObjectRetriever
(_table_retriever) used inside the pipeline, which returns SQLTableSchema objects directly.

Decision logic:
  top similarity >= DB_ROUTE_THRESHOLD  → "db"   (question semantically matches DB schema)
  top similarity <  DB_ROUTE_THRESHOLD  → ""     (no strong DB signal — let LLM decide)
  retriever not ready yet               → ""     (pipeline still initialising — let LLM decide)

We never force-route to "rag" from the vector signal alone — the system prompt and tool
descriptions are the fallback for document questions. Forcing "rag" on low DB score would
incorrectly route general questions ("hello", "what can you help with?") to the RAG service.

The threshold is tunable via DB_ROUTE_THRESHOLD in .env (default 0.25).
Nothing is hardcoded — the signal comes entirely from the live database schema at runtime.
"""
import asyncio
import logging

from ..adapters.db_tool import get_table_retriever
from ..config import settings
from .state import AgentState

logger = logging.getLogger("orchestrator.router")


def _semantic_route(question: str) -> str:
    """Query the raw vector table index and return a routing hint.

    Returns "db" if the question semantically matches the DB schema above the
    configured threshold. Returns "" otherwise so the LLM decides from context.
    """
    retriever = get_table_retriever()
    if retriever is None:
        logger.debug("Table retriever not ready — routing deferred to LLM")
        return ""

    try:
        nodes = retriever.retrieve(question)
        if not nodes:
            logger.info("ROUTE '' (no nodes returned) question=%r", question[:80])
            return ""

        top_score = nodes[0].score if nodes[0].score is not None else 0.0
        top_content = nodes[0].node.get_content()[:60] if hasattr(nodes[0], "node") else str(nodes[0])[:60]
        logger.info(
            "ROUTE top_score=%.3f threshold=%.3f top_match=%r question=%r",
            top_score, settings.DB_ROUTE_THRESHOLD, top_content, question[:80],
        )

        if top_score >= settings.DB_ROUTE_THRESHOLD:
            return "db"
        return ""  # score too low to be confident — let LLM decide

    except Exception as exc:
        logger.warning("Semantic routing error: %s — deferring to LLM", exc)
        return ""


async def routing_node(state: AgentState) -> dict:
    """LangGraph node: run semantic routing once at the start of each conversation turn.

    Runs the vector similarity check in a thread pool so it does not block the event loop.
    Subsequent iterations (after tool calls) skip re-routing — routing_hint is already set.
    """
    if state.get("routing_hint"):
        return {}  # already routed on first pass — no-op on subsequent iterations

    messages = state.get("messages", [])
    if not messages:
        return {"routing_hint": ""}

    last = messages[-1]
    question = (
        last.content
        if hasattr(last, "content") and isinstance(last.content, str)
        else ""
    )

    # Run sync retriever in thread pool — does not block the async event loop
    hint = await asyncio.to_thread(_semantic_route, question)
    logger.info("ROUTING_HINT=%r for question=%r", hint, question[:80])
    return {"routing_hint": hint}

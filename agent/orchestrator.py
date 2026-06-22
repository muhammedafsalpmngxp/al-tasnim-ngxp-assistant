"""
AL TASNIM Agentic AI — LangGraph StateGraph orchestrator.

Flow:
  [START]
    → clarification_gate
        → END (vague query)        — returns clarification question
        → execute_tools            — calls RAG / SQL / analytics per route
            → generate_recommendation  (only for recommend route)
            → validate_and_format      — evidence check + structured answer
                → log_and_return       — observability log + cache write
                    → [END]

Architecture principle (doc v2.1):
  "Use the cheapest, safest, and smallest possible tool path that can
   answer the question correctly."
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore[import]
except ModuleNotFoundError as _lg_err:  # pragma: no cover
    raise ImportError(
        "langgraph is required. Install with: pip install langgraph"
    ) from _lg_err

try:
    import psycopg2  # type: ignore[import]
    _HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    _HAS_PSYCOPG2 = False

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

from .cache import cache_get, cache_put
from .config import human_review_cfg, next_actions_cfg, validation_cfg
from .models import AgentState, RouteType, UserRole
from .router import check_clarification, route_query
from .tools import analytics_tool, rag_tool, recommend_tool, sql_tool
from .validator import (
    check_human_review_needed,
    determine_confidence,
    validate_answer,
)

logger = logging.getLogger(__name__)


# ── Private helpers for node_validate_and_format ──────────────────────────────

def _gather_evidence(
    sql_res:   Optional[Dict[str, Any]],
    rag_res:   Optional[Dict[str, Any]],
    analytics: Optional[Dict[str, Any]],
    rec_text:  Optional[str],
) -> tuple[str, List[str], List[str]]:
    """Combine tool outputs into (combined_answer, evidence, sources)."""
    parts:    List[str] = []
    evidence: List[str] = []
    sources:  List[str] = []

    if sql_res and sql_res.get("answer"):
        parts.append(sql_res["answer"])
        if sql_res.get("sql"):
            evidence.append(f"SQL executed: {sql_res['sql'][:200]}")
        sources += sql_res.get("sources", [])

    if rag_res and rag_res.get("answer"):
        parts.append(rag_res["answer"])
        sources += rag_res.get("sources", [])

    if analytics and analytics.get("summary"):
        parts.append(f"Analytics: {analytics['summary']}")
        evidence.append(analytics["summary"])

    if rec_text:
        parts.append(f"\n{rec_text}")

    combined = "\n\n".join(filter(None, parts)).strip()
    return combined, evidence, sources


def _build_assumptions(route: str) -> List[str]:
    """Return route-appropriate assumption strings."""
    items: List[str] = []
    if route in ("sql", "analytics"):
        items.append(
            "Based on the latest data loaded into the operational database."
        )
    if route == "analytics":
        items.append(
            "Progress metrics assume linear progression from last reported values."
        )
    if route == "recommend":
        items.append(
            "Recommendations based on available data and operational norms"
            " — field conditions may differ."
        )
    return items


def _build_risk_limitation(
    analytics:      Optional[Dict[str, Any]],
    is_valid:       bool,
    invalid_reason: str,
) -> str:
    """Compose the risk/limitation string from analytics metrics and validation."""
    lines: List[str] = []
    if analytics and analytics.get("metrics", {}).get("delay_risk") == "High":
        lines.append("High delay risk detected — immediate action recommended.")
    if not is_valid:
        lines.append(f"Answer validation failed: {invalid_reason}")
    return "; ".join(lines)


def _build_next_action(
    route:      str,
    confidence: str,
    sql_res:    Optional[Dict[str, Any]],
) -> str:
    """Return the recommended next-action string for the given route.
    All text comes from agent_config.yaml next_actions — nothing hardcoded here."""
    na = next_actions_cfg()
    if route == "recommend":
        return na.get("recommend", "")
    if route in ("analytics", "multi") and confidence == "Low":
        return na.get("analytics_low_confidence", "")
    if route == "sql" and not sql_res:
        return na.get("sql_no_data", "")
    return ""


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def node_clarification_gate(state: AgentState) -> Dict[str, Any]:
    """Check for vague queries and determine the route; serve from cache if hit."""
    query = state["query"]
    role  = state.get("user_role", UserRole.OPERATIONS.value)

    cached = cache_get(query, role)
    if cached:
        return {
            **cached,
            "from_cache":           True,
            "clarification_needed": False,
            "route":                cached.get("route", ""),
            "direct_answer":        cached.get("direct_answer", ""),
        }

    needs_clar, clar_text = check_clarification(query)
    if needs_clar:
        return {
            "clarification_needed": True,
            "clarification_text":   clar_text,
            "route":                RouteType.CLARIFICATION.value,
            "direct_answer":        clar_text,
            "confidence":           "N/A",
            "tools_called":         [],
            "sources":              [],
            "evidence":             [],
        }

    determined_route = await route_query(query)
    return {
        "clarification_needed": False,
        "clarification_text":   "",
        "route":                determined_route,
        "from_cache":           False,
    }


async def node_execute_tools(state: AgentState) -> Dict[str, Any]:
    """Call the appropriate tool(s) in parallel where possible."""
    if state.get("from_cache") or state.get("clarification_needed"):
        return {}

    query  = state["query"]
    route  = state.get("route", RouteType.SQL.value)
    top_k  = state.get("top_k", 8)
    called: List[str] = []

    rag_result:       Optional[Dict[str, Any]] = None
    sql_result:       Optional[Dict[str, Any]] = None
    analytics_result: Optional[Dict[str, Any]] = None

    if route == RouteType.RAG.value:
        rag_result = await rag_tool(query, top_k)
        called.append("rag")

    elif route == RouteType.SQL.value:
        sql_result = await sql_tool(query, top_k)
        called.append("sql")

    elif route == RouteType.ANALYTICS.value:
        sql_result = await sql_tool(query, top_k)
        called.append("sql")
        analytics_result = analytics_tool(sql_result, query)
        called.append("analytics")

    elif route == RouteType.MULTI.value:
        sql_result, rag_result = await asyncio.gather(
            sql_tool(query, top_k), rag_tool(query, top_k)
        )
        called.extend(["sql", "rag"])

    elif route == RouteType.RECOMMEND.value:
        sql_result, rag_result = await asyncio.gather(
            sql_tool(query, top_k), rag_tool(query, top_k)
        )
        analytics_result = analytics_tool(sql_result, query)
        called.extend(["sql", "rag", "analytics"])

    else:
        sql_result = await sql_tool(query, top_k)
        called.append("sql")

    return {
        "rag_result":       rag_result,
        "sql_result":       sql_result,
        "analytics_result": analytics_result,
        "tools_called":     called,
    }


async def node_generate_recommendation(state: AgentState) -> Dict[str, Any]:
    """Generate recommendation via Tier-3 LLM. Only runs for the 'recommend' route."""
    if state.get("from_cache") or state.get("clarification_needed"):
        return {}
    if state.get("route") != RouteType.RECOMMEND.value:
        return {}

    rec = await recommend_tool(
        query            = state["query"],
        sql_result       = state.get("sql_result"),
        analytics_result = state.get("analytics_result"),
        rag_result       = state.get("rag_result"),
    )
    return {
        "recommendation": rec.get("recommendation", ""),
        "tools_called":   state.get("tools_called", []) + ["recommend"],
    }


def node_validate_and_format(state: AgentState) -> Dict[str, Any]:
    """Evidence validation + 7-section answer assembly (per architecture doc v2.1)."""
    if state.get("from_cache") or state.get("clarification_needed"):
        return {}

    query     = state["query"]
    route     = state.get("route", "")
    user_role = state.get("user_role", UserRole.OPERATIONS.value)
    sql_res   = state.get("sql_result")
    rag_res   = state.get("rag_result")
    analytics = state.get("analytics_result")
    rec_text  = state.get("recommendation", "")

    combined_answer, evidence, sources = _gather_evidence(
        sql_res, rag_res, analytics, rec_text
    )
    confidence = determine_confidence(route, sql_res, rag_res)

    cfg = validation_cfg()
    is_valid, invalid_reason = validate_answer(
        combined_answer, sources, route, user_role
    )
    if not is_valid:
        combined_answer = cfg.get(
            "refuse_message",
            "I could not find enough verified information."
            " Please rephrase your query.",
        )
        confidence = "Low"

    assumptions    = _build_assumptions(route)
    risk_limitation = _build_risk_limitation(analytics, is_valid, invalid_reason)
    next_action    = _build_next_action(route, confidence, sql_res)

    needs_review, review_reason = check_human_review_needed(
        query, confidence, route, rec_text
    )

    ops_roles = (UserRole.OPERATIONS.value, UserRole.NGXP.value)
    if route == "recommend" and user_role in ops_roles and not needs_review:
        needs_review  = True
        review_reason = human_review_cfg().get(
            "operations_review_message",
            "This recommendation requires operations manager approval.",
        ).strip()

    return {
        "direct_answer":       combined_answer,
        "evidence":            list(set(evidence))[:10],
        "sources":             list(dict.fromkeys(sources))[:10],
        "confidence":          confidence,
        "assumptions":         assumptions,
        "risk_limitation":     risk_limitation,
        "next_action":         next_action,
        "needs_human_review":  needs_review,
        "human_review_reason": review_reason,
    }


async def node_log_and_return(state: AgentState) -> Dict[str, Any]:
    """Write the query trace to agent_query_log and update the cache."""
    if state.get("from_cache"):
        return {}

    _try_db_log(state)

    if not state.get("clarification_needed") and state.get("direct_answer"):
        cache_put(
            state.get("query", ""),
            state.get("user_role", ""),
            {
                "route":           state.get("route", ""),
                "direct_answer":   state.get("direct_answer", ""),
                "evidence":        state.get("evidence", []),
                "sources":         state.get("sources", []),
                "confidence":      state.get("confidence", ""),
                "assumptions":     state.get("assumptions", []),
                "risk_limitation": state.get("risk_limitation", ""),
                "next_action":     state.get("next_action", ""),
            },
        )
    return {}


def _try_db_log(state: AgentState) -> None:
    """Attempt a non-blocking write to agent_query_log. Silently skipped if DB unavailable."""
    if not _HAS_PSYCOPG2 or psycopg2 is None:
        return
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return
    try:
        conn = psycopg2.connect(db_url)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_query_log (
                        id                 SERIAL PRIMARY KEY,
                        session_id         TEXT,
                        user_role          TEXT,
                        query              TEXT,
                        route              TEXT,
                        tools_called       TEXT[],
                        confidence         TEXT,
                        needs_human_review BOOLEAN,
                        from_cache         BOOLEAN,
                        error              TEXT,
                        ts                 TIMESTAMP DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO agent_query_log
                      (session_id, user_role, query, route,
                       tools_called, confidence, needs_human_review,
                       from_cache, error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        state.get("session_id", ""),
                        state.get("user_role", ""),
                        state.get("query", "")[:1000],
                        state.get("route", ""),
                        state.get("tools_called", []),
                        state.get("confidence", ""),
                        state.get("needs_human_review", False),
                        state.get("from_cache", False),
                        state.get("error", ""),
                    ),
                )
        conn.close()
    except (psycopg2.Error, OSError) as exc:
        logger.warning("agent_query_log write failed (non-fatal): %s", exc)


# ── Conditional edge functions ────────────────────────────────────────────────

def _route_after_clarification(
    state: AgentState,
) -> Literal["execute_tools", "__end__"]:
    """Skip to END if clarification needed or response came from cache."""
    if state.get("clarification_needed") or state.get("from_cache"):
        return END
    return "execute_tools"


def _route_after_tools(
    state: AgentState,
) -> Literal["generate_recommendation", "validate_and_format"]:
    """Fan to recommendation node only for the 'recommend' route."""
    if state.get("route") == RouteType.RECOMMEND.value:
        return "generate_recommendation"
    return "validate_and_format"


# ── Build and cache the compiled graph ────────────────────────────────────────

def _build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("clarification_gate",      node_clarification_gate)
    graph.add_node("execute_tools",           node_execute_tools)
    graph.add_node("generate_recommendation", node_generate_recommendation)
    graph.add_node("validate_and_format",     node_validate_and_format)
    graph.add_node("log_and_return",          node_log_and_return)

    graph.add_edge(START, "clarification_gate")
    graph.add_conditional_edges(
        "clarification_gate",
        _route_after_clarification,
        {"execute_tools": "execute_tools", END: END},
    )
    graph.add_conditional_edges(
        "execute_tools",
        _route_after_tools,
        {
            "generate_recommendation": "generate_recommendation",
            "validate_and_format":     "validate_and_format",
        },
    )
    graph.add_edge("generate_recommendation", "validate_and_format")
    graph.add_edge("validate_and_format",     "log_and_return")
    graph.add_edge("log_and_return",          END)

    return graph.compile()


# Mutable container — avoids module-level `global` statement
_GRAPH: Dict[str, Any] = {"instance": None}


def get_graph():
    """Return the compiled LangGraph instance (singleton, built once)."""
    if _GRAPH["instance"] is None:
        _GRAPH["instance"] = _build_graph()
    return _GRAPH["instance"]


# ── Public API ────────────────────────────────────────────────────────────────

async def run_agent(
    query:      str,
    session_id: str = "default",
    user_role:  str = UserRole.OPERATIONS.value,
    top_k:      int = 8,
) -> Dict[str, Any]:
    """Run the full agent pipeline and return the final state dict."""
    t0 = time.monotonic()

    initial_state: AgentState = {
        "query":               query,
        "session_id":          session_id,
        "user_role":           user_role,
        "top_k":               top_k,
        "route":               "",
        "clarification_needed": False,
        "clarification_text":  "",
        "rag_result":          None,
        "sql_result":          None,
        "analytics_result":    None,
        "recommendation":      None,
        "tools_called":        [],
        "evidence":            [],
        "sources":             [],
        "direct_answer":       "",
        "confidence":          "Medium",
        "assumptions":         [],
        "risk_limitation":     "",
        "next_action":         "",
        "needs_human_review":  False,
        "human_review_reason": "",
        "from_cache":          False,
        "error":               "",
        "latency_ms":          0,
    }

    graph = get_graph()
    try:
        final_state = await graph.ainvoke(initial_state)
    except (RuntimeError, ValueError, asyncio.TimeoutError) as exc:
        logger.exception("Agent pipeline error: %s", exc)
        final_state = {
            **initial_state,
            "direct_answer": "An internal error occurred. Please try again.",
            "error":         str(exc),
            "confidence":    "Low",
        }

    final_state["latency_ms"] = int((time.monotonic() - t0) * 1000)
    return final_state

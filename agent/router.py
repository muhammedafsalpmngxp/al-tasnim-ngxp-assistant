"""
Policy-based query router — deterministic first, cheap LLM only as fallback.

Routing tiers (per architecture doc v2.1):
  Tier 1 — Rule-based regex matching (zero cost)
  Tier 2 — Small/cheap LLM classification (used only when rules are unclear)

Route types:
  rag           → document knowledge only
  sql           → live operational database only
  analytics     → database + precomputed analytics
  multi         → database + documents
  recommend     → database + analytics + documents + LLM recommendation
  clarification → query is too vague; ask user for more detail
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional, Tuple

try:
    from ollama import AsyncClient as OllamaAsyncClient
except ImportError:
    OllamaAsyncClient = None  # type: ignore[assignment,misc]

from .config import clarification_cfg, llm_cfg, llm_fallback_model, prompts_cfg, router_cfg
from .models import RouteType

logger = logging.getLogger(__name__)


# ── Precompile regex patterns from config ─────────────────────────────────────

def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


def _load_patterns() -> dict[str, list[re.Pattern]]:
    cfg = router_cfg()
    return {
        "rag_priority": _compile_patterns(cfg.get("rag_priority_patterns", [])),
        "rag":          _compile_patterns(cfg.get("rag_patterns",       [])),
        "sql":          _compile_patterns(cfg.get("sql_patterns",       [])),
        "analytics":    _compile_patterns(cfg.get("analytics_patterns", [])),
        "multi":        _compile_patterns(cfg.get("multi_patterns",     [])),
        "recommend":    _compile_patterns(cfg.get("recommend_patterns", [])),
    }


def _load_clarification() -> Tuple[list[re.Pattern], re.Pattern]:
    cfg = clarification_cfg()
    vague = _compile_patterns(cfg.get("vague_patterns", []))
    specific = re.compile(
        cfg.get("specific_id_pattern", r"\b\d{4,6}\b"),
        re.IGNORECASE,
    )
    return vague, specific


# Singleton state — stored in a mutable dict so no `global` statement is needed
_STATE: dict = {
    "patterns":       {},
    "vague_patterns": [],
    "specific_pat":   None,
}


def _ensure_loaded() -> None:
    if not _STATE["patterns"]:
        _STATE["patterns"]       = _load_patterns()
        vague, specific          = _load_clarification()
        _STATE["vague_patterns"] = vague
        _STATE["specific_pat"]   = specific


def _matches_any(query: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(query) for p in patterns)


# ── Clarification gate ────────────────────────────────────────────────────────

def check_clarification(query: str) -> Tuple[bool, str]:
    """
    Returns (needs_clarification, clarification_text).
    If the query matches a vague pattern AND contains no specific identifier,
    return True with an appropriate clarification prompt.
    """
    _ensure_loaded()
    cfg = clarification_cfg()

    is_vague   = _matches_any(query, _STATE["vague_patterns"])
    has_id     = bool(_STATE["specific_pat"] and _STATE["specific_pat"].search(query))
    word_count = len(query.split())

    if is_vague and not has_id and word_count < 10:
        prompts = cfg.get("prompts", {})
        q_lower = query.lower()
        if "finish" in q_lower or "complet" in q_lower:
            text = prompts.get("when_will_finish", prompts.get("default", ""))
        elif "delay" in q_lower or "behind" in q_lower:
            text = prompts.get("why_delayed", prompts.get("default", ""))
        elif "update" in q_lower:
            text = prompts.get("give_me_update", prompts.get("default", ""))
        else:
            text = prompts.get("default", "Please provide more specific details.")
        return True, text.strip()

    return False, ""


# ── Tier-1 rule-based router ─────────────────────────────────────────────────

def rule_based_route(query: str) -> Optional[str]:
    """
    Return a route string if any pattern matches, otherwise None.

    Priority order (highest → lowest specificity):
      recommend > multi > analytics > rag_priority > sql > rag
    rag_priority patterns override sql so that "FLAF procedure" → RAG, not SQL.
    """
    _ensure_loaded()
    patterns = _STATE["patterns"]

    # Ordered list of (pattern_key, route_value) — first match wins.
    # Two entries map to RAG: rag_priority (before sql) and rag (fallback).
    priority: list[tuple[str, str]] = [
        ("recommend",    RouteType.RECOMMEND),
        ("multi",        RouteType.MULTI),
        ("analytics",    RouteType.ANALYTICS),
        ("rag_priority", RouteType.RAG),
        ("sql",          RouteType.SQL),
        ("rag",          RouteType.RAG),
    ]
    return next(
        (route for key, route in priority if _matches_any(query, patterns[key])),
        None,
    )


# ── Tier-2 LLM-based router ───────────────────────────────────────────────────

_VALID_ROUTES = (
    {r.value for r in RouteType}
    - {RouteType.CLARIFICATION.value, RouteType.UNKNOWN.value}
)


async def llm_based_route(query: str) -> str:
    """Call the small classifier LLM when rule-based routing is inconclusive."""
    if OllamaAsyncClient is None:
        logger.warning("ollama not installed — defaulting route to 'sql'")
        return RouteType.SQL.value

    # System prompt loaded from config — no hardcoded strings in Python
    system_prompt = prompts_cfg().get("llm_router_system", "")
    cfg        = llm_cfg("classifier")
    model_name = (
        os.environ.get("LLM_MODEL") or cfg.get("model", "") or llm_fallback_model()
    )
    ollama_url = (
        os.environ.get("OLLAMA_URL")
        or cfg.get("base_url", "http://localhost:11434")
    )
    try:
        client = OllamaAsyncClient(host=ollama_url)
        resp   = await client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
        )
        raw   = resp["message"]["content"].strip().lower()
        route = raw.split()[0] if raw else ""
        return route if route in _VALID_ROUTES else RouteType.SQL.value
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        logger.warning("LLM router failed, defaulting to 'sql': %s", exc)
        return RouteType.SQL.value


# ── Public entry point ────────────────────────────────────────────────────────

async def route_query(query: str) -> str:
    """
    Determine the best route for a query.
    Tries rule-based first (free); falls back to cheap LLM only if unclear.
    """
    route = rule_based_route(query)
    if route:
        logger.debug("Rule-based route: %s → %s", query[:60], route)
        return route

    logger.debug("No rule match — calling LLM classifier for: %s", query[:60])
    route = await llm_based_route(query)
    logger.debug("LLM-based route: %s → %s", query[:60], route)
    return route

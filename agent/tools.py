"""
Tool implementations for the AL TASNIM Agentic AI module.

Each tool is an async function that calls the upstream RAG/SQL service
(prod_rag.py on port 8000) and returns a normalised dict.

Tools available:
  rag_tool         — document knowledge retrieval (PDF, Excel, KT transcripts)
  sql_tool         — structured operational data from PostgreSQL
  analytics_tool   — compute planned vs actual, rate, and delay risk from SQL rows
  recommend_tool   — generate structured recommendations via Tier-3 LLM

The LLM must NEVER directly execute SQL.
All database access goes through prod_rag.py's safe SQL layer.
All prompt strings, model names, and URLs come from agent_config.yaml via config.py.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

try:
    import httpx as _httpx  # type: ignore[import]
    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False

try:
    from ollama import AsyncClient as _OllamaClient  # type: ignore[import]
    _HAS_OLLAMA = True
except ImportError:  # pragma: no cover
    _OllamaClient = None  # type: ignore[assignment]
    _HAS_OLLAMA = False

try:
    from groq import AsyncGroq as _AsyncGroq  # type: ignore[import]
    _HAS_GROQ = True
except ImportError:
    _AsyncGroq = None  # type: ignore[assignment]
    _HAS_GROQ = False

from .config import analytics_cfg, llm_cfg, llm_fallback_model, prompts_cfg, rag_service_cfg

logger = logging.getLogger(__name__)


# ── HTTP client ───────────────────────────────────────────────────────────────

def _rag_client():
    """Build an httpx async client pointed at the RAG service."""
    if not _HAS_HTTPX or _httpx is None:
        raise ImportError("httpx is required: pip install httpx")
    cfg = rag_service_cfg()
    return _httpx.AsyncClient(
        base_url=cfg["base_url"],
        timeout=cfg.get("timeout_seconds", 120),
    )


# ── RAG tool ──────────────────────────────────────────────────────────────────

async def rag_tool(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Retrieve answers from indexed documents via prod_rag.py /search."""
    cfg = rag_service_cfg()
    url = cfg.get("search_endpoint", "/search")
    try:
        async with _rag_client() as client:
            resp = await client.post(url, json={"query": query, "top_k": top_k})
            resp.raise_for_status()
            data = resp.json()
            return {
                "tool":    "rag",
                "answer":  data.get("answer", ""),
                "sources": data.get("sources", []),
                "chunks":  data.get("results", []),
                "error":   None,
            }
    except (_httpx.HTTPError, _httpx.RequestError, OSError) as exc:
        logger.error("rag_tool error: %s", exc)
        return {
            "tool": "rag", "answer": "", "sources": [],
            "chunks": [], "error": str(exc),
        }


# ── SQL tool ──────────────────────────────────────────────────────────────────

async def sql_tool(query: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Query operational data from PostgreSQL via the safe SQL compiler in prod_rag.py.
    Calls /ask; the upstream service routes to SQL or RAG internally.
    """
    cfg = rag_service_cfg()
    url = cfg.get("ask_endpoint", "/ask")
    try:
        async with _rag_client() as client:
            resp = await client.post(
                url,
                json={
                    "query":      query,
                    "session_id": "agent_sql",
                    "top_k":      top_k,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "tool":       "sql",
                "answer":     data.get("answer", ""),
                "rows":       data.get("rows", []),
                "sql":        data.get("sql", ""),
                "sources":    data.get("sources", []),
                "query_type": data.get("query_type", ""),
                "error":      None,
            }
    except (_httpx.HTTPError, _httpx.RequestError, OSError) as exc:
        logger.error("sql_tool error: %s", exc)
        return {
            "tool": "sql", "answer": "", "rows": [], "sql": "",
            "sources": [], "query_type": "", "error": str(exc),
        }


# ── Analytics tool ────────────────────────────────────────────────────────────

def _safe_float(val: Any) -> Optional[float]:
    """Cast val to float, returning None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _progress_from_row(row: Dict[str, Any]) -> Optional[float]:
    """Extract the best-available progress value from a SQL result row.
    Column priority order comes from agent_config.yaml analytics.progress_columns."""
    for col in analytics_cfg().get("progress_columns", ["over_all_progress_percentages", "progress"]):
        val = _safe_float(row.get(col))
        if val is not None:
            return val
    return None


def analytics_tool(sql_result: Dict[str, Any], _query: str = "") -> Dict[str, Any]:
    """
    Compute planned-vs-actual, rate, and delay-risk metrics from SQL rows.
    Runs locally — no LLM call or network request.
    """
    rows: List[Dict[str, Any]] = sql_result.get("rows", [])
    if not rows:
        return {
            "tool": "analytics", "metrics": {},
            "summary": "No data to analyse.", "error": None,
        }

    progress_vals = [_progress_from_row(r) for r in rows]
    progress_vals = [v for v in progress_vals if v is not None]

    acfg = analytics_cfg()
    completed_threshold  = acfg.get("completed_threshold",    100)
    behind_threshold     = acfg.get("behind_threshold",        50)
    risk_high_pct        = acfg.get("delay_risk_high_pct",    0.5)
    risk_medium_pct      = acfg.get("delay_risk_medium_pct",  0.25)

    metrics: Dict[str, Any] = {}
    if progress_vals:
        total = len(progress_vals)
        metrics["avg_progress"]    = round(sum(progress_vals) / total, 2)
        metrics["max_progress"]    = round(max(progress_vals), 2)
        metrics["min_progress"]    = round(min(progress_vals), 2)
        metrics["total_wells"]     = total
        metrics["completed_wells"] = sum(1 for v in progress_vals if v >= completed_threshold)
        metrics["behind_wells"]    = sum(1 for v in progress_vals if v < behind_threshold)
        behind_pct = metrics["behind_wells"] / max(total, 1)
        metrics["delay_risk"] = (
            "High" if behind_pct > risk_high_pct
            else "Medium" if behind_pct > risk_medium_pct
            else "Low"
        )

    lines: List[str] = []
    if "avg_progress" in metrics:
        lines.append(
            f"Average progress: {metrics['avg_progress']}%"
            f" across {metrics['total_wells']} items"
        )
        lines.append(
            f"Completed ({completed_threshold}%): {metrics['completed_wells']}"
            f" | Behind (<{behind_threshold}%): {metrics['behind_wells']}"
        )
        lines.append(f"Delay risk: {metrics.get('delay_risk', 'Unknown')}")

    return {
        "tool":    "analytics",
        "metrics": metrics,
        "summary": " | ".join(lines) if lines else "Analytics computed.",
        "error":   None,
    }


# ── Recommendation tool ───────────────────────────────────────────────────────

def _build_recommend_prompt(
    query:            str,
    sql_result:       Optional[Dict[str, Any]],
    analytics_result: Optional[Dict[str, Any]],
    rag_result:       Optional[Dict[str, Any]],
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) from config templates."""
    cfg = prompts_cfg()
    system_tmpl   = cfg.get("recommend_system", "")
    user_tmpl     = cfg.get("recommend_template", "")

    sql_summary       = (sql_result or {}).get("answer", "No SQL data.")
    analytics_summary = (analytics_result or {}).get("summary", "No analytics.")
    rag_context       = (rag_result or {}).get("answer", "No document context.")

    user_prompt = user_tmpl.format(
        query             = query,
        sql_summary       = sql_summary[:800],
        analytics_summary = analytics_summary[:400],
        rag_context       = rag_context[:800],
    )
    return system_tmpl, user_prompt


async def _call_groq(
    system_prompt: str,
    user_prompt:   str,
) -> Optional[str]:
    """Call Groq Tier-3 model. Returns text or None on failure."""
    if not _HAS_GROQ or _AsyncGroq is None:
        return None
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return None
    cfg = llm_cfg("reasoning")
    try:
        client = _AsyncGroq(api_key=groq_key)
        resp   = await client.chat.completions.create(
            model       = cfg.get("model", "llama-3.3-70b-versatile"),
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = cfg.get("max_tokens", 2048),
            temperature = cfg.get("temperature", 0.1),
        )
        return (resp.choices[0].message.content or "").strip()
    except (ValueError, RuntimeError, AttributeError, KeyError) as exc:
        logger.warning("Groq recommendation failed, falling back to ollama: %s", exc)
        return None


async def _call_ollama(
    system_prompt: str,
    user_prompt:   str,
) -> str:
    """Call Ollama Tier-2 generator model. Returns text or fallback message."""
    if not _HAS_OLLAMA or _OllamaClient is None:
        return (
            "Unable to generate recommendation at this time."
            " Please review the SQL and analytics data manually."
        )
    cfg        = llm_cfg("generator")
    model_name = (
        os.environ.get("LLM_MODEL") or cfg.get("model", "") or llm_fallback_model()
    )
    ollama_url = (
        os.environ.get("OLLAMA_URL")
        or cfg.get("base_url", "http://localhost:11434")
    )
    try:
        client = _OllamaClient(host=ollama_url)
        resp   = await client.chat(
            model    = model_name,
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return resp["message"]["content"].strip()
    except (OSError, RuntimeError, KeyError) as exc:
        logger.error("recommend_tool ollama error: %s", exc)
        return (
            "Unable to generate recommendation at this time."
            " Please review the SQL and analytics data manually."
        )


async def recommend_tool(
    query:            str,
    sql_result:       Optional[Dict[str, Any]],
    analytics_result: Optional[Dict[str, Any]],
    rag_result:       Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate structured recommendations using the Tier-3 reasoning LLM.
    Prompts are loaded from agent_config.yaml (prompts section).
    Falls back to Ollama if GROQ_API_KEY is not set.
    """
    system_prompt, user_prompt = _build_recommend_prompt(
        query, sql_result, analytics_result, rag_result
    )

    text = await _call_groq(system_prompt, user_prompt)
    if text is None:
        text = await _call_ollama(system_prompt, user_prompt)

    return {"tool": "recommend", "recommendation": text, "error": None}

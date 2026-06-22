"""
Load agent_config.yaml with environment variable substitution.
All agent configuration is read from here — nothing hardcoded in Python.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


_CONFIG_PATH = Path(__file__).parent.parent / "config" / "agent_config.yaml"


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR_NAME} placeholders with values from os.environ."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r"\$\{([^}]+)\}", _replace, text)


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """Load, env-substitute, and cache the full agent_config.yaml."""
    raw = _CONFIG_PATH.read_text()
    substituted = _substitute_env_vars(raw)
    return yaml.safe_load(substituted)


# ── Convenience accessors ──────────────────────────────────────────────────


def server_cfg() -> Dict[str, Any]:
    """Return the server block (host, port, title, version)."""
    return get_config()["server"]


def rag_service_cfg() -> Dict[str, Any]:
    """Return the upstream RAG/SQL service connection block."""
    return get_config()["rag_service"]


def llm_cfg(tier: str) -> Dict[str, Any]:
    """Return LLM config for the given tier: 'classifier' | 'generator' | 'reasoning'."""
    return get_config()["llm"][tier]


def router_cfg() -> Dict[str, Any]:
    """Return the query-routing regex patterns block."""
    return get_config()["router"]


def clarification_cfg() -> Dict[str, Any]:
    """Return the clarification-gate patterns and prompts."""
    return get_config()["clarification"]


def validation_cfg() -> Dict[str, Any]:
    """Return the evidence-validation rules (min length, hallucination phrases, etc.)."""
    return get_config()["validation"]


def answer_format_cfg() -> Dict[str, Any]:
    """Return the required answer-format sections list."""
    return get_config()["answer_format"]


def human_review_cfg() -> Dict[str, Any]:
    """Return the human-review trigger patterns and confidence threshold."""
    return get_config()["human_review"]


def cache_cfg() -> Dict[str, Any]:
    """Return the cache settings (enabled, TTL, semantic threshold)."""
    return get_config()["cache"]


def logging_cfg() -> Dict[str, Any]:
    """Return the observability / query-log settings."""
    return get_config()["logging"]


def embeddings_cfg() -> Dict[str, Any]:
    """Return the embedding model settings (model name, device)."""
    return get_config()["embeddings"]


def prompts_cfg() -> Dict[str, Any]:
    """Return all LLM prompt templates (recommend_system, recommend_template, etc.)."""
    return get_config()["prompts"]


def llm_fallback_model() -> str:
    """Return the absolute last-resort model name from config (used when env var and tier model are both empty)."""
    return get_config()["llm"].get("fallback_model", "llama3.1:8b")

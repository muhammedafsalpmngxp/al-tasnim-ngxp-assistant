"""
In-memory semantic cache for the agent.
Phase 1: exact-match + cosine similarity using sentence-transformers.
Phase 2: upgrade to Redis (swap this module only).

Cache key: SHA256 hash of (normalised query, user_role).
Semantic check: cosine similarity of query embedding vs stored embeddings.
TTL: configured in agent_config.yaml (default 300 seconds).
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import os

import numpy as np

from .config import cache_cfg, embeddings_cfg

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AgentCache:
    """Thread-safe in-memory cache with exact and semantic hit support."""

    def __init__(self) -> None:
        cfg = cache_cfg()
        self._enabled   = cfg.get("enabled", True)
        self._threshold = cfg.get("semantic_threshold", 0.92)
        self._ttl       = cfg.get("ttl_seconds", 300)

        # {sha256_key: (response_dict, expire_time)}
        self._exact: Dict[str, Tuple[Dict[str, Any], float]] = {}

        # [(embedding, response_dict, expire_time)]
        self._semantic: List[Tuple[np.ndarray, Dict[str, Any], float]] = []

        self._embed_model = None  # lazy init
        embed_cfg = embeddings_cfg()
        self._embed_model_name = (
            os.environ.get("EMBED_MODEL")
            or embed_cfg.get("model", "")
            or embed_cfg.get("default_model", "all-MiniLM-L6-v2")
        )
        self._embed_device = embed_cfg.get("device", "cpu")

    def _load_embedder(self):
        """Lazy-init the sentence-transformer model from config."""
        if self._embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer(
                    self._embed_model_name, device=self._embed_device
                )
            except Exception as e:
                logger.warning("Semantic cache disabled — embedder error: %s", e)
        return self._embed_model

    def _embed(self, text: str) -> Optional[np.ndarray]:
        model = self._load_embedder()
        if model is None:
            return None
        try:
            return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        except Exception:
            return None

    def _cache_key(self, query: str, user_role: str) -> str:
        return _sha256(f"{user_role}||{query.lower().strip()}")

    def get(self, query: str, user_role: str) -> Optional[Dict[str, Any]]:
        if not self._enabled:
            return None
        now = time.time()
        key = self._cache_key(query, user_role)

        # Exact match
        if key in self._exact:
            payload, exp = self._exact[key]
            if now < exp:
                logger.debug("Cache HIT (exact): %s", query[:60])
                return payload
            del self._exact[key]

        # Semantic match
        emb = self._embed(query)
        if emb is not None:
            self._semantic = [(e, r, t) for e, r, t in self._semantic if now < t]
            for stored_emb, response, _ in self._semantic:
                sim = _cosine(emb, stored_emb)
                if sim >= self._threshold:
                    logger.debug("Cache HIT (semantic, sim=%.3f): %s", sim, query[:60])
                    return response

        return None

    def put(self, query: str, user_role: str, response: Dict[str, Any]) -> None:
        if not self._enabled:
            return
        exp = time.time() + self._ttl
        key = self._cache_key(query, user_role)
        self._exact[key] = (response, exp)

        emb = self._embed(query)
        if emb is not None:
            self._semantic.append((emb, response, exp))

        logger.debug("Cache SET: %s", query[:60])


# Module-level singleton
_cache = AgentCache()


def cache_get(query: str, user_role: str) -> Optional[Dict[str, Any]]:
    return _cache.get(query, user_role)


def cache_put(query: str, user_role: str, response: Dict[str, Any]) -> None:
    _cache.put(query, user_role, response)

"""
retriever.py — BM25, Dense, and Hybrid table retrievers for NL2SQL.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings as LlamaSettings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.indices.loading import load_index_from_storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Merge multiple ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, item in enumerate(ranked_list, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25TableRetriever:
    """Sparse keyword retriever using BM25Okapi over table context strings."""

    def __init__(self, table_contexts: dict[str, str]) -> None:
        self.table_names: list[str] = list(table_contexts.keys())
        tokenized = [ctx.lower().split() for ctx in table_contexts.values()]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return the top-k table names ranked by BM25 score."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self.table_names), key=lambda x: x[0], reverse=True)
        return [name for _, name in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Dense retriever — VectorStoreIndex only, no ObjectIndex
# ---------------------------------------------------------------------------

class DenseTableRetriever:
    """
    Dense semantic retriever using VectorStoreIndex directly.

    Uses hash-based cache to reduce startup from ~3.5 min to ~5 sec.
    Cache auto-invalidates when db-schema.yaml changes or INDEX_CACHE_VERSION changes.
    """

    def __init__(
        self,
        table_contexts: dict[str, str],
        embed_model,
        cache_dir: Optional[str] = None,
        schema_yaml_path: Optional[str] = None,
        cache_version: str = "1.0.0",
    ) -> None:
        self.table_names = list(table_contexts.keys())
        self._cache_dir = cache_dir
        self._schema_yaml_path = schema_yaml_path
        self._cache_version = cache_version

        # Set global embed_model ONCE permanently — required at both build and query time
        LlamaSettings.embed_model = embed_model
        logger.info("Global embed_model set: %s", embed_model.__class__.__name__)

        # Try loading from cache first
        if cache_dir and schema_yaml_path:
            loaded = self._load_index_from_cache(cache_dir, schema_yaml_path)
            if loaded is not None:
                self.index = loaded
                logger.info("Dense retriever loaded from cache: %s", cache_dir)
                return

        # Build from scratch
        logger.info("Building dense index from scratch (%d tables)…", len(table_contexts))
        documents = [
            Document(text=context, metadata={"table_name": table_name})
            for table_name, context in table_contexts.items()
        ]
        self.index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)

        # Persist cache
        if cache_dir and schema_yaml_path:
            self._save_index_to_cache(cache_dir, schema_yaml_path, cache_version)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _schema_hash(self, schema_yaml_path: str) -> str:
        with open(schema_yaml_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _hash_file(self, cache_dir: str) -> Path:
        return Path(cache_dir) / "schema_hash.txt"

    def _version_file(self, cache_dir: str) -> Path:
        return Path(cache_dir) / "cache_version.txt"

    def _disk_space_info(self, path: str) -> str:
        try:
            usage = shutil.disk_usage(Path(path).parent)
            return f"{usage.free / (1024 * 1024):.0f} MB free"
        except Exception:
            return "unknown"

    def _permission_check(self, path: str) -> str:
        test = Path(path) / ".write_test"
        try:
            test.touch()
            test.unlink()
            return "OK"
        except Exception:
            return "permission denied"

    def _load_index_from_cache(
        self, cache_dir: str, schema_yaml_path: str
    ) -> Optional[VectorStoreIndex]:
        cache_path = Path(cache_dir)

        # All five files must exist — partial writes are treated as cache miss
        required = [
            cache_path / "vector_store.json",
            cache_path / "docstore.json",
            cache_path / "index_store.json",
            self._hash_file(cache_dir),
            self._version_file(cache_dir),
        ]
        missing = [str(f) for f in required if not f.exists()]
        if missing:
            logger.info("Cache miss — missing files: %s", missing)
            return None

        # Hash check
        saved_hash = self._hash_file(cache_dir).read_text().strip()
        current_hash = self._schema_hash(schema_yaml_path)
        if saved_hash != current_hash:
            logger.info(
                "Cache miss — schema changed (saved=%s…, current=%s…)",
                saved_hash[:8], current_hash[:8],
            )
            return None

        # Version check
        saved_version = self._version_file(cache_dir).read_text().strip()
        if saved_version != self._cache_version:
            logger.info(
                "Cache miss — version changed (saved=%s, current=%s)",
                saved_version, self._cache_version,
            )
            return None

        try:
            storage_context = StorageContext.from_defaults(persist_dir=cache_dir)
            index = load_index_from_storage(storage_context=storage_context)
            logger.info("VectorStoreIndex loaded from %s", cache_dir)
            return index
        except Exception as exc:
            logger.warning("Cache load failed (%s); will rebuild from scratch", exc)
            return None

    def _save_index_to_cache(
        self, cache_dir: str, schema_yaml_path: str, cache_version: str
    ) -> bool:
        try:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            self.index.storage_context.persist(persist_dir=cache_dir)

            current_hash = self._schema_hash(schema_yaml_path)
            self._hash_file(cache_dir).write_text(current_hash)
            self._version_file(cache_dir).write_text(cache_version)

            total_kb = sum(
                f.stat().st_size for f in cache_path.iterdir() if f.is_file()
            ) / 1024
            logger.info(
                "Cache saved to %s (%.1f KB, hash=%s…, version=%s)",
                cache_dir, total_kb, current_hash[:8], cache_version,
            )
            return True

        except Exception as exc:
            logger.warning(
                "CACHE SAVE FAILED: %s | dir=%s | disk=%s | permissions=%s | "
                "WARNING: index will NOT persist — next restart rebuilds from scratch (3.5 min).",
                exc, cache_dir,
                self._disk_space_info(cache_dir),
                self._permission_check(cache_dir),
            )
            return False

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return the top-k table names ranked by embedding similarity."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        table_names = []
        for r in results:
            name = r.metadata.get("table_name")
            if name:
                table_names.append(name)
            else:
                logger.warning("Result missing table_name metadata: %s", r)
        return table_names


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridTableRetriever:
    """Combines BM25 and dense retrieval via Reciprocal Rank Fusion."""

    def __init__(self, bm25: BM25TableRetriever, dense: DenseTableRetriever, top_k: int) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.top_k = top_k

    def retrieve(self, query: str) -> list[str]:
        """Return top_k table names fused from BM25 and dense rankings."""
        candidate_k = self.top_k * 2
        bm25_ranked = self.bm25.retrieve(query, candidate_k)
        dense_ranked = self.dense.retrieve(query, candidate_k)
        fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
        return fused[: self.top_k]

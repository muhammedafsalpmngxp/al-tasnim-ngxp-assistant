"""
retriever.py — BM25, Dense, and Hybrid table retrievers for NL2SQL.
"""
from __future__ import annotations

from rank_bm25 import BM25Okapi

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core import SQLDatabase


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    score(d) = sum_over_lists( 1 / (k + rank(d)) )
    """
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
        # Pair (score, name) and sort descending
        ranked = sorted(
            zip(scores, self.table_names),
            key=lambda x: x[0],
            reverse=True,
        )
        return [name for _, name in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

class DenseTableRetriever:
    """Dense semantic retriever using LlamaIndex VectorStoreIndex over table schemas."""

    def __init__(
        self,
        table_schemas: list[SQLTableSchema],
        sql_database: SQLDatabase,
        embed_model,
    ) -> None:
        node_mapping = SQLTableNodeMapping(sql_database)
        self.obj_index = ObjectIndex.from_objects(
            table_schemas,
            node_mapping,
            VectorStoreIndex,
            embed_model=embed_model,
        )

    def retrieve(self, query: str, top_k: int) -> list[str]:
        """Return the top-k table names ranked by embedding similarity."""
        retriever = self.obj_index.as_retriever(similarity_top_k=top_k)
        # ObjectIndex retriever returns the actual objects (SQLTableSchema), not NodeWithScore
        results = retriever.retrieve(query)
        table_names: list[str] = []
        for r in results:
            # Results may be SQLTableSchema objects or NodeWithScore wrappers
            if isinstance(r, SQLTableSchema):
                table_names.append(r.table_name)
            elif hasattr(r, "node"):
                meta = r.node.metadata or {}
                name = meta.get("table_name") or getattr(r.node, "table_name", None)
                if name:
                    table_names.append(name)
            elif hasattr(r, "table_name"):
                table_names.append(r.table_name)
        return table_names


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridTableRetriever:
    """
    Combines BM25 and dense retrieval via Reciprocal Rank Fusion.

    Each sub-retriever is queried with top_k * 2 candidates to give RRF
    enough material; the final list is trimmed to top_k.
    """

    def __init__(
        self,
        bm25: BM25TableRetriever,
        dense: DenseTableRetriever,
        top_k: int,
    ) -> None:
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

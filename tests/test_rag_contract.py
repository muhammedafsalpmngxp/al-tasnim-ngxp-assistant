"""Verify the RAG /search contract via the skeleton's search() function."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag_assistant.src.server import search


def test_rag_search_contract():
    r = search("casing running procedure", 5)
    assert r.success is True and isinstance(r.passages, list)
    print("PASS — RAG /search contract:", r.model_dump())


if __name__ == "__main__":
    test_rag_search_contract()

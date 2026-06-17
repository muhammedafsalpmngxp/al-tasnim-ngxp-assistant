"""Verify both tools register correctly (no provider libs needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import get_tools, tool_names


def test_tools_registered():
    names = set(tool_names())
    assert {"query_database", "search_documents"} <= names, f"missing tools: {names}"
    assert all(hasattr(t, "name") for t in get_tools())
    print("PASS — registered tools:", sorted(names))


if __name__ == "__main__":
    test_tools_registered()

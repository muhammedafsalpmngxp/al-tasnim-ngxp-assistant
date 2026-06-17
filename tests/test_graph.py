"""Offline wiring test — proves tool results feed back and the loop terminates.

Runnable both ways:  pytest  |  PYTHONPATH=. python tests/test_graph.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.graph.builder import build_app


@tool
def echo(text: str) -> str:
    """Echo helper for testing."""
    return '{"status": "ok", "echo": "%s"}' % text


class _FakeLLM:
    def __init__(self):
        self.turn = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.turn += 1
        if self.turn == 1:
            return AIMessage(content="", tool_calls=[
                {"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"}])
        assert any(isinstance(m, ToolMessage) for m in messages), \
            "tool result was NOT fed back to the LLM"
        return AIMessage(content="Rig is operational. Sources: echo tool.")


def test_graph_feedback_loop():
    app = build_app(llm=_FakeLLM(), tools=[echo])
    result = app.invoke(
        {"messages": [HumanMessage(content="say hi")], "iterations": 0},
        {"configurable": {"thread_id": "t1"}},
    )
    msgs = result["messages"]
    assert any(isinstance(m, ToolMessage) for m in msgs), "no ToolMessage in history"
    assert msgs[-1].content.startswith("Rig is operational")
    assert result["iterations"] <= 3
    print("PASS — flow:", [type(m).__name__ for m in msgs])


if __name__ == "__main__":
    test_graph_feedback_loop()

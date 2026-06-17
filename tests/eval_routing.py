"""Routing eval harness — measures tool-selection accuracy against a labeled set.

Run against a live model:  PYTHONPATH=. python tests/eval_routing.py
It binds the real tools to the configured LLM and checks which tool(s) the model
chooses for each question (it does NOT execute the tools). Tracks accuracy so you
can protect the >90% routing KPI as you tune prompts/descriptions.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage

from src.prompts import SYSTEM_PROMPT
from src.llm import get_chat_model
from src.adapters import get_tools

# (question, expected set of tool names)
CASES = [
    ("What is the status of rig 104?", {"query_database"}),
    ("How many wells are currently active?", {"query_database"}),
    ("What is the casing-running procedure?", {"search_documents"}),
    ("Define spud date.", {"search_documents"}),
    ("When will Well 30750 finish and what's the completion checklist?",
     {"query_database", "search_documents"}),
]


def chosen_tools(llm, question):
    msg = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)])
    return {tc["name"] for tc in (msg.tool_calls or [])}


def main():
    llm = get_chat_model().bind_tools(get_tools())
    passed = 0
    for q, expected in CASES:
        got = chosen_tools(llm, q)
        ok = got == expected
        passed += ok
        print(f"[{'OK ' if ok else 'XX '}] {q}\n      expected={sorted(expected)} got={sorted(got)}")
    print(f"\nRouting accuracy: {passed}/{len(CASES)} = {100*passed/len(CASES):.0f}%")


if __name__ == "__main__":
    main()

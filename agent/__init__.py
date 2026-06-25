"""
agent/ — Intelligent routing and processing layer for the DB assistant.

Components:
  router.py    — LLM-based SIMPLE/COMPLEX classifier
  agent.py     — FunctionAgent for complex multi-step queries (per-request, isolated)
  tools.py     — execute_sql, get_schema, compute_stats FunctionTools
  clarifier.py — Pre-processing: asks user if question is too vague
  memory.py    — Context injection: enriches question with conversation history
  decomposer.py — Splits multi-part questions into focused sub-questions
  validator.py  — Post-processing: flags uncertain/off-topic answers
"""

from agent.router import route_question
from agent.agent import ask_complex_agent
from agent.clarifier import check_clarification
from agent.memory import inject_context, build_context_string
from agent.decomposer import decompose_question
from agent.validator import validate_answer

__all__ = [
    "route_question",
    "ask_complex_agent",
    "check_clarification",
    "inject_context",
    "build_context_string",
    "decompose_question",
    "validate_answer",
]

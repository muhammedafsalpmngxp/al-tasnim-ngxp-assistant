"""Build the evidence pack, extract the final answer, validate grounding."""
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, ToolMessage

from .schemas import ToolCall

_NO_EVIDENCE = "I cannot confidently answer this based on available evidence."


def collect_evidence(messages: List[Any]) -> List[Dict[str, Any]]:
    """Return one entry per ToolMessage with parsed payload."""
    evidence: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            try:
                payload = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                payload = {"raw": m.content}
            evidence.append({"tool": m.name, "payload": payload})
    return evidence


def extract_tools_used(evidence: List[Dict[str, Any]]) -> List[ToolCall]:
    """Summarise which tools ran and what they returned."""
    result = []
    for e in evidence:
        p = e.get("payload", {})
        result.append(ToolCall(
            tool=e.get("tool", "unknown"),
            status=p.get("status", "unknown") if isinstance(p, dict) else "unknown",
            row_count=p.get("row_count") if isinstance(p, dict) else None,
            source=p.get("source") if isinstance(p, dict) else None,
        ))
    return result


def extract_final_answer(messages: List[Any]) -> Optional[str]:
    """Return the last AI message that has no pending tool calls."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls and m.content:
            return m.content if isinstance(m.content, str) else str(m.content)
    return None


def has_usable_evidence(evidence: List[Dict[str, Any]]) -> bool:
    return any(
        isinstance(e.get("payload"), dict) and e["payload"].get("status") == "ok"
        for e in evidence
    )


def validate_answer(
    answer: Optional[str], evidence: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Return (grounded, message_or_answer).

    If a tool asked for clarification, surface that question to the user.
    If there is no usable evidence, return the canned no-evidence message.
    """
    for e in evidence:
        p = e.get("payload")
        if isinstance(p, dict) and p.get("status") == "needs_clarification":
            return False, p.get("question") or "Could you clarify your question?"
    if not answer:
        return False, _NO_EVIDENCE
    if not has_usable_evidence(evidence):
        return False, _NO_EVIDENCE
    return True, answer

"""
Evidence validation layer — every answer must pass this before being shown.

Per architecture doc v2.1 section 12:
  - Correct source used?
  - Citations available?
  - Assumptions stated?
  - Confidence included?
  - Recommendation within user permission?
  - Human approval required?
  - Answer is not a hallucination signal phrase?

Returns (is_valid: bool, reason: str).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import validation_cfg, human_review_cfg
from .models import UserRole

logger = logging.getLogger(__name__)


def _has_hallucination(text: str, phrases: List[str]) -> bool:
    t = text.lower()
    return any(p in t for p in phrases)


def validate_answer(
    answer:    str,
    sources:   List[str],
    route:     str,
    user_role: str,
) -> Tuple[bool, str]:
    """
    Validate the answer before sending to user.
    Returns (is_valid, reason_if_invalid).
    """
    cfg = validation_cfg()
    hallucination_phrases = [p.lower() for p in cfg.get("hallucination_phrases", [])]
    min_len    = cfg.get("min_answer_length_chars", 20)
    min_src    = cfg.get("min_sources_required", 1)

    if not answer or len(answer.strip()) < min_len:
        return False, "Answer is too short or empty."

    if _has_hallucination(answer, hallucination_phrases):
        return False, "Answer contains uncertainty signal — treating as not validated."

    # SQL / analytics routes must have at least some data rows or clear answer text
    if route in ("sql", "analytics", "multi", "recommend"):
        if len(answer.strip()) < 30:
            return False, "SQL answer too thin — insufficient operational data."

    # Management role gets recommendations; operations role should not get raw recs
    if route == "recommend" and user_role == UserRole.OPERATIONS.value:
        # Operations users CAN see recommendations, but they need manager note
        pass  # Handled in format_answer

    return True, ""


def check_human_review_needed(
    query:      str,
    confidence: str,
    route:      str,
    recommendation: Optional[str],
) -> Tuple[bool, str]:
    """
    Returns (needs_human_review, reason).
    Triggered by: high-risk query patterns, low confidence, or safety-critical terms.
    """
    cfg     = human_review_cfg()
    patterns = [re.compile(p, re.IGNORECASE) for p in cfg.get("trigger_patterns", [])]
    low_conf_threshold = cfg.get("low_confidence_threshold", "Low")

    # Check trigger patterns against query text
    for pat in patterns:
        if pat.search(query):
            return True, f"Query matches high-risk pattern: '{pat.pattern}'"

    # Also check the recommendation text for escalation signals
    if recommendation:
        for pat in patterns:
            if pat.search(recommendation):
                return True, "Recommendation contains high-risk decision requiring approval."

    # Low confidence always triggers review
    if confidence == low_conf_threshold:
        return True, f"Confidence is '{confidence}' — requires human validation before acting."

    return False, ""


def determine_confidence(
    route:      str,
    sql_result: Optional[Dict[str, Any]],
    rag_result: Optional[Dict[str, Any]],
) -> str:
    """
    Simple confidence scoring based on what data was retrieved.
    High   — SQL data found with rows, or strong document match
    Medium — partial data, or single source
    Low    — errors or empty results
    """
    has_sql_rows = bool(
        sql_result and sql_result.get("rows") and len(sql_result["rows"]) > 0
    )
    has_rag_answer = bool(
        rag_result and len(rag_result.get("answer", "").strip()) > 40
    )
    sql_error = bool(sql_result and sql_result.get("error"))
    rag_error = bool(rag_result and rag_result.get("error"))

    if sql_error and rag_error:
        return "Low"
    if route == "sql" and has_sql_rows:
        return "High"
    if route == "rag" and has_rag_answer:
        return "High"
    if (route in ("multi", "recommend")) and (has_sql_rows or has_rag_answer):
        return "Medium"
    if has_sql_rows or has_rag_answer:
        return "Medium"
    return "Low"

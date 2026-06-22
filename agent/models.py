"""
Pydantic request/response models + LangGraph AgentState TypedDict.
All I/O contracts live here — no business logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    NGXP       = "ngxp"          # internal dev team
    OPERATIONS = "operations"     # field operations users
    MANAGEMENT = "management"     # supervisors / managers
    ADMIN      = "admin"          # system administrators


class RouteType(str, Enum):
    RAG           = "rag"           # document knowledge only
    SQL           = "sql"           # database query only
    ANALYTICS     = "analytics"     # SQL + precomputed analytics
    MULTI         = "multi"         # SQL + RAG combined
    RECOMMEND     = "recommend"     # SQL + Analytics + RAG + LLM recommendation
    CLARIFICATION = "clarification" # query too vague — ask user for details
    UNKNOWN       = "unknown"       # fallback


# ── API request / response ───────────────────────────────────────────────────

class AgentRequest(BaseModel):
    query:      str
    session_id: str      = Field(default="default", description="Conversation session ID")
    user_role:  UserRole = Field(default=UserRole.OPERATIONS)
    top_k:      int      = Field(default=8, ge=1, le=50)


class AgentResponse(BaseModel):
    query:                  str
    direct_answer:          str
    evidence:               List[str] = []
    sources:                List[str] = []
    confidence:             str = "Medium"
    assumptions:            List[str] = []
    risk_limitation:        Optional[str] = None
    recommended_next_action: Optional[str] = None
    route:                  str = ""
    tools_called:           List[str] = []
    needs_human_review:     bool = False
    human_review_reason:    Optional[str] = None
    from_cache:             bool = False
    session_id:             str = ""
    clarification_needed:   bool = False
    clarification_text:     Optional[str] = None
    error:                  Optional[str] = None
    latency_ms:             int = 0


# ── LangGraph AgentState ──────────────────────────────────────────────────────
# Every field that a node might write must appear here.
# Nodes return a dict containing ONLY the fields they update; LangGraph merges
# the returned dict into the running state automatically.

class AgentState(TypedDict, total=False):
    # ── Input (set once at entry) ─────────────────────────────────────────────
    query:       str
    session_id:  str
    user_role:   str
    top_k:       int

    # ── Router ────────────────────────────────────────────────────────────────
    route:                str
    clarification_needed: bool
    clarification_text:   str

    # ── Tool results ──────────────────────────────────────────────────────────
    rag_result:         Optional[Dict[str, Any]]
    sql_result:         Optional[Dict[str, Any]]
    analytics_result:   Optional[Dict[str, Any]]
    recommendation:     Optional[str]
    tools_called:       List[str]

    # ── Evidence pack ─────────────────────────────────────────────────────────
    evidence:  List[str]
    sources:   List[str]

    # ── Answer ────────────────────────────────────────────────────────────────
    direct_answer: str
    confidence:    str
    assumptions:   List[str]
    risk_limitation: str
    next_action:   str

    # ── Escalation ────────────────────────────────────────────────────────────
    needs_human_review:  bool
    human_review_reason: str

    # ── Meta ──────────────────────────────────────────────────────────────────
    from_cache: bool
    error:      str
    latency_ms: int

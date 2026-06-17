"""Request / response schemas (Pydantic v2)."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000, description="Natural-language question")
    session_id: Optional[str] = Field(default="default", description="Conversation thread ID")
    user_id: Optional[str] = Field(default="anonymous", description="Optional caller identifier")

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("question must be at least 2 non-space characters")
        return v


class ToolCall(BaseModel):
    tool: str
    status: str                        # "ok" | "error" | "needs_clarification"
    row_count: Optional[int] = None
    source: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    tools_used: List[ToolCall] = Field(default_factory=list)
    iterations: int = 0
    session_id: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0

from typing import Any, List, Optional
from pydantic import BaseModel


class FinalResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    data: Optional[List[dict]] = None
    sql: Optional[str] = None
    row_count: int = 0
    tables_used: List[str] = []
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0

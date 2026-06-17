from typing import Optional, List, Any, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum

class Operator(str, Enum):
    EQ = "="
    NE = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    IN = "in"
    LIKE = "like"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

class AggregateFunc(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"

class Filter(BaseModel):
    col: str
    op: Operator
    value: Any = None
    resolved_value: Any = None
    resolution_status: str = ""
    
    @field_validator('op', mode='before')
    @classmethod
    def validate_op(cls, v):
        if isinstance(v, str):
            v = v.lower()
        if v not in [op.value for op in Operator]:
            raise ValueError(f"Invalid operator: {v}")
        return v

class Aggregate(BaseModel):
    func: AggregateFunc
    group_by: Optional[List[str]] = Field(default_factory=list)

class OrderBy(BaseModel):
    col: str
    direction: Literal["ASC", "DESC"] = "ASC"

class Intent(BaseModel):
    table: str = ""
    columns: List[str] = Field(default_factory=list)
    filters: List[Filter] = Field(default_factory=list)
    aggregate: Optional[Aggregate] = None
    order_by: List[OrderBy] = Field(default_factory=list)
    limit: int = Field(default=100, ge=1)
    natural_language: str = ""

    @field_validator("table")
    @classmethod
    def validate_table(cls, v):
        from .config import settings
        if v and v not in settings.ALLOWED_TABLES:
            raise ValueError(f"Table '{v}' not in allowed catalog")
        return v

    @field_validator("limit")
    @classmethod
    def cap_limit(cls, v):
        from .config import settings
        return min(v, settings.MAX_ROWS)

class ConversationState(BaseModel):
    session_id: str
    pending_clarification: bool = False
    clarification_asked: Optional[str] = None
    original_query: Optional[str] = None
    previous_intent: Optional[Intent] = None
    query_history: List[str] = Field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3

class IntentResponse(BaseModel):
    success: bool
    intent: Optional[Intent] = None
    error: Optional[str] = None
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    suggestions: Optional[List[Any]] = None

class FinalResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    data: Optional[List[dict]] = None
    sql: Optional[str] = None
    row_count: int = 0
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    suggestions: Optional[List[str]] = None
    error: Optional[str] = None
    execution_time_ms: float = 0

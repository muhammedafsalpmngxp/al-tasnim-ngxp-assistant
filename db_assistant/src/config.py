from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Database - from env only, NO DEFAULTS
    DB_SERVER: str
    DB_NAME: str
    DB_READONLY_USER: str
    DB_READONLY_PASSWORD: str

    # LLM provider: "gemini" (Google) or "local" (Ollama)
    LLM_PROVIDER: str = "local"
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

    # Local Ollama settings
    MODEL_NAME: str = "qwen3:8b"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 800
    LLM_TIMEOUT_SEC: int = 120

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def normalize_llm_provider(cls, v):
        if isinstance(v, str):
            v = v.split("#")[0].strip().lower()
        if v not in ("gemini", "local"):
            raise ValueError("LLM_PROVIDER must be 'gemini' or 'local'")
        return v
    
    # Limits
    MAX_ROWS: int = 1000
    QUERY_TIMEOUT_SEC: int = 30
    POOL_SIZE: int = 10
    
    # Value resolution
    FUZZY_THRESHOLD: int = 80
    AMBIGUITY_MARGIN: int = 10
    
    # Tables - policy (acceptable hardcoding)
    ALLOWED_TABLES: List[str] = [
        "2026_Well_Delivery_Scope_Well_Type",
        "ActivityTaskPlan",
        "WMR",
        "Employee",
        "crews",
        "CrewEmployee",
        "Equipment",
        "task_daily",
        "Revenue",
        "SAP_DRILLING_SEQUENCE"
    ]
    
    # PII patterns - does NOT block 'name' columns
    PII_PATTERNS: List[str] = [
        r'email', r'mail', r'contact', r'phone', r'gsm', r'mobile',
        r'assignee', r'supervisor_email', r'manager_email',
        r'address', r'ph\s*name', r'permit\s*applicant'
    ]
    
    # Snapshot tables
    SNAPSHOT_TABLES: List[str] = ["WMR", "Job_Progress_PlanSnapshot"]
    SNAPSHOT_DATE_COLUMNS: dict = {
        "WMR": "Week_Number",
        "Job_Progress_PlanSnapshot": "SnapshotMonth"
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "db_assistant.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

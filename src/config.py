"""Central configuration with startup validation.

Nothing is hardcoded elsewhere — everything reads from here, overridable via env
or a .env file. Validation runs at import so misconfiguration fails fast and loud.
"""
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider (pluggable) ---
    # Accepts "ollama", "local" (alias for ollama), "gemini", or "groq".
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "qwen3:8b"                    # used when LLM_PROVIDER=local/ollama
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"    # used when LLM_PROVIDER=gemini
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"  # used when LLM_PROVIDER=groq
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SEC: int = 120
    LLM_MAX_TOKENS: int = 1024
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # --- DB tool (imported in-process; reads creds from the single root .env) ---
    DB_ENV_FILE: str = ".env"
    DB_TOOL_TIMEOUT: float = 120.0

    # --- RAG tool (HTTP service) ---
    RAG_TOOL_URL: str = "http://localhost:8002"
    RAG_TOOL_TIMEOUT: float = 60.0
    RAG_TOP_K: int = 5
    RAG_RETRIES: int = 2

    # --- Orchestrator ---
    MAX_ITERATIONS: int = 5
    RECURSION_LIMIT: int = 12
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    RELOAD: bool = False

    # --- Semantic routing ---
    # Minimum cosine similarity for the table vector index to count as a DB match.
    # Raise to require stronger match before routing to DB; lower to be more inclusive.
    DB_ROUTE_THRESHOLD: float = 0.25

    # --- Observability ---
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None          # set a path to also write logs to a file
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "al-tasnim-orchestrator"

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def _normalize_provider(cls, v):
        if isinstance(v, str):
            v = v.split("#")[0].strip().lower()
        if v == "local":
            v = "ollama"
        if v not in ("ollama", "gemini", "groq"):
            raise ValueError("LLM_PROVIDER must be one of: ollama, local, gemini, groq")
        return v

    @model_validator(mode="after")
    def _validate(self):
        if self.LLM_PROVIDER == "gemini" and not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        if self.LLM_PROVIDER == "gemini" and self.GOOGLE_API_KEY:
            key = self.GOOGLE_API_KEY
            # Google AI Studio API keys always start with "AIza".
            # OAuth2 tokens (AQ., ya29., etc.) are NOT valid here.
            if not key.startswith("AIza"):
                import warnings
                warnings.warn(
                    "GOOGLE_API_KEY does not look like a Google AI Studio API key "
                    "(expected prefix 'AIza'). OAuth2 / service-account tokens are not "
                    "supported. Get a valid key at https://aistudio.google.com/apikey",
                    stacklevel=2,
                )
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if not (1 <= self.PORT <= 65535):
            raise ValueError(f"PORT out of range: {self.PORT}")
        if self.MAX_ITERATIONS < 1:
            raise ValueError("MAX_ITERATIONS must be >= 1")
        if self.RECURSION_LIMIT < 2:
            raise ValueError("RECURSION_LIMIT must be >= 2")
        return self


settings = Settings()

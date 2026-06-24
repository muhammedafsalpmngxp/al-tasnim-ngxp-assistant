"""
config.py — Application settings loaded from .env via pydantic-settings.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DB_SERVER: str = "20.98.112.250"
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_DRIVER: str = "ODBC Driver 17 for SQL Server"

    # LLM Provider selection
    LLM_PROVIDER: str = "groq"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_TEMPERATURE: float = 0.0

    # Google Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"

    # Ollama / local
    LLM_MODEL: str = "qwen2.5:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embeddings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    # Schema
    SCHEMA_YAML_PATH: str = "data/db-schema.yaml"

    # Index cache — reduces startup from ~3.5 min to ~5 sec after first run
    # Increment INDEX_CACHE_VERSION to force a full rebuild without changing the schema
    INDEX_CACHE_DIR: str = "index_cache"
    INDEX_CACHE_VERSION: str = "1.0.0"

    # Retrieval
    TABLE_TOP_K: int = 5

    # Pipeline
    MAX_SQL_RETRIES: int = 2

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance
settings = Settings()

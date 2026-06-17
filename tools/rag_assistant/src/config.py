from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    TOP_K: int = 5
    LOG_LEVEL: str = "INFO"
    VECTOR_DB_URL: Optional[str] = None
    EMBEDDING_MODEL: str = "nomic-embed-text"


settings = Settings()

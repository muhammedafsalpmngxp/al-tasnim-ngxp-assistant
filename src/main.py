"""Entry point: python -m src.main"""
import uvicorn

from .config import settings
from .observability import configure_logging

if __name__ == "__main__":
    configure_logging()
    uvicorn.run("src.server:app", host=settings.HOST, port=settings.PORT, reload=settings.RELOAD)

"""Structured logging with correlation IDs + optional file + LangSmith tracing."""
import logging
import os
from contextvars import ContextVar

from .config import settings

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")
_FORMAT = "%(asctime)s %(levelname)s [cid=%(correlation_id)s] %(name)s: %(message)s"


class _CorrelationFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_var.get()
        return True


def configure_logging():
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    fmt = logging.Formatter(_FORMAT)
    handlers = [logging.StreamHandler()]
    if settings.LOG_FILE:
        handlers.append(logging.FileHandler(settings.LOG_FILE))
    for h in handlers:
        h.addFilter(_CorrelationFilter())
        h.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger(__name__).info(
        "Logging configured (level=%s, file=%s)", settings.LOG_LEVEL, settings.LOG_FILE or "-")


def configure_tracing():
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        logging.getLogger(__name__).info(
            "LangSmith tracing enabled (project=%s)", settings.LANGSMITH_PROJECT)

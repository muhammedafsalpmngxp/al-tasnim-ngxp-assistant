"""Shared utility helpers for the database assistant."""

import logging
import re
from typing import Optional

REASONING_TAG_PATTERN = re.compile(
    r"\x3cthink\x3e.*?\x3c/think\x3e",
    re.DOTALL | re.IGNORECASE,
)

def strip_reasoning_tags(text: str) -> str:
    return REASONING_TAG_PATTERN.sub("", text).strip()

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging for CLI scripts."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

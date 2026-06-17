"""Shared utility helpers for the database assistant."""

import logging
import re
from typing import Optional

# Matches <think>...</think> blocks (including newlines) produced by reasoning models.
# Using literal tag characters (not hex escapes) for readability.
_REASONING_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning_tags(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks from LLM output and strip whitespace."""
    if not text:
        return ""
    return _REASONING_TAG_PATTERN.sub("", text).strip()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logging for CLI scripts."""
    handlers: list = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

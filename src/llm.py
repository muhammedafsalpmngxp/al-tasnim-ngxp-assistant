"""Pluggable LLM factory (native tool-calling; no prompt/JSON parsing)."""
import logging

from .config import settings

logger = logging.getLogger("orchestrator.llm")

_KNOWN_TOOL_MODELS = (
    "qwen2.5", "qwen3", "llama3.1", "llama3.2", "llama3.3",
    "mistral", "mixtral", "firefunction", "command-r", "hermes",
)


def _check_tool_support():
    if settings.LLM_PROVIDER == "ollama":
        m = settings.LLM_MODEL.lower()
        if not any(k in m for k in _KNOWN_TOOL_MODELS):
            logger.warning(
                "Model %r is not in the known tool-calling list; the agent cannot "
                "route if this model lacks tool support. Known: %s",
                settings.LLM_MODEL, ", ".join(_KNOWN_TOOL_MODELS),
            )


def get_chat_model():
    provider = settings.LLM_PROVIDER
    _check_tool_support()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        logger.info("LLM provider=ollama model=%s", settings.LLM_MODEL)
        return ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=settings.LLM_MAX_TOKENS,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("LLM provider=gemini model=%s", settings.LLM_MODEL)
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

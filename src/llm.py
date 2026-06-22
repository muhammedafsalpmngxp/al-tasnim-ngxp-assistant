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


class _GeminiChatModel:
    """Thin wrapper around ChatGoogleGenerativeAI that converts 401/auth errors
    into clear, actionable messages instead of a raw Google API traceback."""

    def __init__(self, **kwargs):
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._inner = ChatGoogleGenerativeAI(**kwargs)

    # Delegate every attribute access so LangChain sees the real model object.
    def __getattr__(self, name):
        return getattr(self._inner, name)

    def invoke(self, *args, **kwargs):
        return self._call_with_auth_guard(self._inner.invoke, *args, **kwargs)

    def ainvoke(self, *args, **kwargs):
        return self._call_with_auth_guard(self._inner.ainvoke, *args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        # Return a wrapper so bind_tools() result also gets auth-guard coverage.
        bound = self._inner.bind_tools(*args, **kwargs)
        wrapper = _GeminiChatModel.__new__(_GeminiChatModel)
        wrapper._inner = bound
        return wrapper

    @staticmethod
    def _call_with_auth_guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "UNAUTHENTICATED" in msg or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in msg:
                logger.error(
                    "Gemini authentication failed (401). "
                    "Your GOOGLE_API_KEY is invalid or is an OAuth2 token. "
                    "Get a valid API key at https://aistudio.google.com/apikey and set it as GOOGLE_API_KEY in .env"
                )
                raise RuntimeError(
                    "Gemini authentication failed: invalid GOOGLE_API_KEY. "
                    "Get a valid key at https://aistudio.google.com/apikey"
                ) from exc
            raise


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
        key = settings.GOOGLE_API_KEY or ""
        if not key.startswith("AIza"):
            logger.error(
                "GOOGLE_API_KEY looks invalid (got prefix %r). "
                "Google AI Studio keys start with 'AIza'. "
                "Get a valid key at https://aistudio.google.com/apikey",
                key[:6] if key else "(empty)",
            )
        logger.info("LLM provider=gemini model=%s", settings.GEMINI_MODEL_NAME)
        return _GeminiChatModel(
            model=settings.GEMINI_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_output_tokens=settings.LLM_MAX_TOKENS,
            google_api_key=key,
        )
    if provider == "groq":
        from langchain_groq import ChatGroq
        logger.info("LLM provider=groq model=%s", settings.GROQ_MODEL_NAME)
        return ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.GROQ_API_KEY,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

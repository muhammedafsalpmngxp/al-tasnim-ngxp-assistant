"""LLM client abstraction — supports Gemini (cloud) and local Ollama."""

import logging
from typing import Dict, Any, Optional, List

from .config import settings
from .utils import strip_reasoning_tags

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for Gemini and local Ollama."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.timeout = settings.LLM_TIMEOUT_SEC
        self._client = None
        self._available: Optional[bool] = None
        self._init_client()

    def _init_client(self):
        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_local()

    def _init_gemini(self):
        try:
            import google.generativeai as genai

            if not settings.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set in .env")

            genai.configure(api_key=settings.GOOGLE_API_KEY)

            # Log available models to help diagnose API key / region issues
            try:
                available = [
                    m.name for m in genai.list_models()
                    if "generateContent" in m.supported_generation_methods
                ]
                logger.info("Available Gemini models: %s", available)
            except Exception as list_err:
                logger.warning("Could not list Gemini models (check API key): %s", list_err)

            self._client = genai.GenerativeModel(
                settings.GEMINI_MODEL_NAME,
                generation_config={
                    "temperature": settings.TEMPERATURE,
                    "max_output_tokens": settings.MAX_TOKENS,
                },
            )
            logger.info("Gemini client initialized with model: %s", settings.GEMINI_MODEL_NAME)
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini: {e}")

    def _init_local(self):
        try:
            import ollama

            self._client = ollama.Client(timeout=self.timeout)
            logger.info("Ollama client initialized with model: %s", settings.MODEL_NAME)
        except ImportError:
            raise ImportError("ollama not installed. Run: pip install ollama")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Ollama: {e}")

    def chat(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        if self.provider == "gemini":
            return self._chat_gemini(messages, **kwargs)
        return self._chat_local(messages, **kwargs)

    def _chat_gemini(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        system_prompt = ""
        user_message = ""

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")

        full_prompt = f"{system_prompt}\n\n{user_message}".strip()
        request_options = {"timeout": self.timeout}

        response = self._client.generate_content(full_prompt, request_options=request_options)
        raw = strip_reasoning_tags(response.text or "")

        return {
            "message": {"content": raw},
            "provider": "gemini",
        }

    def _chat_local(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        options = dict(kwargs.get("options", {}))
        # Do NOT disable thinking — qwen3 and similar models return empty when think=False.
        # strip_reasoning_tags() removes <think>...</think> blocks from the output.

        response = self._client.chat(
            model=settings.MODEL_NAME,
            messages=messages,
            options=options,
        )

        raw = strip_reasoning_tags(response["message"]["content"] or "")

        return {
            "message": {"content": raw},
            "provider": "local",
        }

    def check_available(self) -> bool:
        if self._available is True:
            return True  # cache success only; failures are re-checked each call

        try:
            if self.provider == "gemini":
                import google.generativeai as genai
                next(genai.list_models(), None)
            else:
                import ollama
                ollama.list()
            self._available = True
        except Exception as e:
            logger.error("Provider check failed: %s", e)
            self._available = False

        return self._available


_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

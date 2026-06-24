"""
llm_factory.py — Factory functions for LLM and embedding model creation.
"""
from __future__ import annotations


def get_llm(settings):
    """Return the appropriate LLM instance based on settings.LLM_PROVIDER."""
    provider = settings.LLM_PROVIDER.lower()

    if provider == "groq":
        from llama_index.llms.groq import Groq

        return Groq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.GROQ_TEMPERATURE,
        )

    if provider == "gemini":
        from llama_index.llms.gemini import Gemini

        model_name = settings.GEMINI_MODEL_NAME
        if not model_name.startswith(("models/", "tunedModels/")):
            model_name = f"models/{model_name}"
        return Gemini(
            model_name=model_name,
            api_key=settings.GOOGLE_API_KEY,
        )

    if provider == "local":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=settings.LLM_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            request_timeout=120.0,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}'. "
        "Valid options: 'groq', 'gemini', 'local'."
    )


def get_embed_model(settings):
    """Return a HuggingFaceEmbedding instance configured from settings."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE,
    )

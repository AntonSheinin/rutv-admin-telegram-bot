from __future__ import annotations

from app.config import Settings
from app.llm import LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        from app.openai_llm import OpenAILLMClient

        return OpenAILLMClient(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

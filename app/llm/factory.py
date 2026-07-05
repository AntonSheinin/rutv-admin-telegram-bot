from __future__ import annotations

from app.core.config import Settings
from app.llm.client import LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "openai":
        from app.llm.openai import OpenAILLMClient

        return OpenAILLMClient(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

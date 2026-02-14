"""LLM service abstraction for answer generation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class BaseLLMService(ABC):
    """Interface for LLM answer generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text response from model prompt."""


class OpenAILLMService(BaseLLMService):
    """OpenAI chat completion implementation."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM provider")
        self.client = OpenAI(api_key=settings.openai_api_key)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.openai_llm_model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful retrieval assistant. Answer using provided context only. "
                        "If information is missing, explicitly say you could not find it in the documents."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or "No response generated."


class LocalRuleBasedLLMService(BaseLLMService):
    """Fallback local service for environments without external API access."""

    def generate(self, prompt: str) -> str:
        return (
            "Local fallback mode is active. Please configure OPENAI_API_KEY and "
            "LLM_PROVIDER=openai for production-quality responses.\n\n"
            f"Prompt preview:\n{prompt[:1000]}"
        )


def get_llm_service() -> BaseLLMService:
    """Factory returning configured LLM provider."""
    if settings.llm_provider.lower() == "openai":
        return OpenAILLMService()
    return LocalRuleBasedLLMService()

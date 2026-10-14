"""LLM access through any OpenAI-compatible chat-completions endpoint.

The provider is configuration only (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL); nothing here is
specific to one vendor. Tests inject a scripted fake that implements `LLMClient`.
"""

import logging
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import openai
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user"
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    model_name: str

    async def complete(self, messages: list[ChatMessage]) -> LLMResult: ...


class LLMUnavailableError(AppError):
    status_code = 502
    code = "LLM_UNAVAILABLE"


class LLMNotConfiguredError(AppError):
    status_code = 503
    code = "LLM_NOT_CONFIGURED"


class OpenAICompatibleClient:
    def __init__(self, settings: Settings, *, http_client: httpx.AsyncClient | None = None) -> None:
        if not settings.llm_configured:
            raise LLMNotConfiguredError(
                "Question answering is not configured: set LLM_BASE_URL, LLM_API_KEY and "
                "LLM_MODEL on the server."
            )
        assert settings.llm_api_key and settings.llm_model
        self.model_name = settings.llm_model
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_s,
            max_retries=1,
            http_client=http_client,
        )

    async def complete(self, messages: list[ChatMessage]) -> LLMResult:
        extra = {"seed": self._settings.llm_seed} if self._settings.llm_seed is not None else {}
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],  # type: ignore[misc]
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_output_tokens,
                **extra,
            )
        except openai.APIError as exc:
            # Details (status, provider message) go to the logs only, never to the client.
            logger.warning("LLM request failed", extra={"error": f"{type(exc).__name__}: {exc}"})
            raise LLMUnavailableError(
                "The answering service is temporarily unavailable. Please try again."
            ) from exc
        if not response.choices or response.choices[0].message.content is None:
            raise LLMUnavailableError("The answering service returned an empty response.")
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content,
            model=response.model or self.model_name,
            usage=(
                {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
                if usage
                else {}
            ),
        )

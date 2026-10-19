"""One OpenAI-compatible client per backend, with the large tier's offline fallback."""

import logging
from dataclasses import dataclass
from typing import Literal

import httpx
from openai import APIError, AsyncOpenAI

from app.config import Settings, TierConfig

log = logging.getLogger(__name__)


@dataclass
class ProviderResult:
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    model: str
    tier: str


class ProviderError(Exception):
    """Raised when no tier can serve the request."""


class Providers:
    def __init__(self, settings: Settings, tiers: dict[str, TierConfig]):
        self.tiers = tiers
        # max_retries=0: the fallback logic below is ours, not the SDK's.
        # An explicit httpx client keeps one HTTP stack across the app (and lets respx mock it).
        self.http = httpx.AsyncClient(timeout=60)
        self.ollama = AsyncOpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",
            max_retries=0,
            http_client=self.http,
        )
        self.groq = (
            AsyncOpenAI(
                base_url=settings.groq_base_url,
                api_key=settings.groq_api_key,
                max_retries=0,
                http_client=self.http,
            )
            if settings.groq_api_key
            else None
        )

    def _client(self, tier: str) -> AsyncOpenAI | None:
        return self.groq if self.tiers[tier].provider == "groq" else self.ollama

    async def _call(
        self,
        tier: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> ProviderResult:
        client = self._client(tier)
        if client is None:
            raise ProviderError(f"{tier}_not_configured")
        kwargs: dict = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        model = self.tiers[tier].model
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        choice = resp.choices[0]
        usage = resp.usage
        return ProviderResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=model,
            tier=tier,
        )

    async def complete(
        self,
        tier: Literal["small", "large"],
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> ProviderResult:
        args = (messages, temperature, max_tokens, json_mode)
        if tier == "small":
            try:
                return await self._call("small", *args)
            except APIError as e:
                log.warning("small tier failed: %s", e)
                raise ProviderError("small_unavailable") from e
        try:
            return await self._call("large", *args)
        except (APIError, ProviderError) as e:
            log.warning("large tier failed, using large_offline: %s", e)
        try:
            return await self._call("large_offline", *args)
        except APIError as e:
            log.warning("large_offline tier failed: %s", e)
            raise ProviderError("all_tiers_unavailable") from e

    async def aclose(self) -> None:
        await self.http.aclose()

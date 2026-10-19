import json
from pathlib import Path

import httpx
import pytest
import respx

from app.config import Settings, load_tiers
from app.providers import ProviderError, Providers

OLLAMA = "http://localhost:11434/v1/chat/completions"
GROQ = "https://api.groq.com/openai/v1/chat/completions"
TIERS = load_tiers(Path("config/models.yaml"))
MSGS = [{"role": "user", "content": "hi"}]


def completion(model: str, text: str = "hello") -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


def make(groq_key: str | None = "gsk_test") -> Providers:
    settings = Settings(_env_file=None, database_url="x", redis_url="x", groq_api_key=groq_key)
    return Providers(settings, TIERS)


@respx.mock
async def test_large_uses_groq():
    respx.post(GROQ).respond(200, json=completion("llama-3.3-70b-versatile"))
    result = await make().complete("large", MSGS, 0.0, None, False)
    assert result.tier == "large"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)


@respx.mock
async def test_large_falls_back_on_429():
    respx.post(GROQ).respond(429, json={"error": {"message": "slow down"}})
    respx.post(OLLAMA).respond(200, json=completion("llama3.1:8b"))
    result = await make().complete("large", MSGS, 0.0, None, False)
    assert result.tier == "large_offline"
    assert result.model == "llama3.1:8b"


@respx.mock
async def test_large_falls_back_without_key():
    groq = respx.post(GROQ).respond(200, json=completion("llama-3.3-70b-versatile"))
    respx.post(OLLAMA).respond(200, json=completion("llama3.1:8b"))
    result = await make(groq_key=None).complete("large", MSGS, 0.0, None, False)
    assert groq.called is False
    assert result.tier == "large_offline"


@respx.mock
async def test_small_connection_error_raises():
    respx.post(OLLAMA).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as exc:
        await make().complete("small", MSGS, 0.0, None, False)
    assert str(exc.value) == "small_unavailable"


@respx.mock
async def test_everything_down_raises():
    respx.post(GROQ).respond(503, json={"error": {"message": "down"}})
    respx.post(OLLAMA).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ProviderError) as exc:
        await make().complete("large", MSGS, 0.0, None, False)
    assert str(exc.value) == "all_tiers_unavailable"


@respx.mock
async def test_json_mode_sends_response_format():
    groq = respx.post(GROQ).respond(200, json=completion("llama-3.3-70b-versatile", "{}"))
    await make().complete("large", MSGS, 0.0, None, True)
    body = json.loads(groq.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}

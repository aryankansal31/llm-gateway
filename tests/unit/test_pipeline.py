import hashlib
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import fakeredis
import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncOpenAI

from app.config import Settings, load_prices
from app.db import RequestLogEntry
from app.main import Deps, create_app
from app.providers import ProviderError, ProviderResult
from app.router import RouteDecision

KEY = "gw_testkey"
AUTH = {"Authorization": f"Bearer {KEY}"}
LOG_COLUMNS = [f.name for f in fields(RequestLogEntry)]


class FakePool:
    """Stands in for asyncpg: knows one API key and records request_log inserts."""

    def __init__(self):
        self.key_id = uuid4()
        self.logs: list[tuple] = []

    async def fetchval(self, sql, key_hash):
        return self.key_id if key_hash == hashlib.sha256(KEY.encode()).hexdigest() else None

    async def execute(self, sql, *args):
        self.logs.append(args)


class FakeProviders:
    def __init__(self, small="fake answer", large="fake answer"):
        self.behaviour = {"small": small, "large": large}
        self.calls: list[str] = []

    async def complete(self, tier, messages, temperature, max_tokens, json_mode):
        self.calls.append(tier)
        outcome = self.behaviour[tier]
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResult(outcome, "stop", 10, 5, f"{tier}-model", tier)


class FakeRouter:
    def decide(self, requested_model, messages):
        if requested_model in ("small", "large"):
            return RouteDecision(requested_model, None, "forced")
        return RouteDecision("small", 0.9, "model")


class FakeCache:
    def __init__(self, fail=False):
        self.store: dict = {}
        self.puts = 0
        self.fail = fail

    async def get(self, messages, params_hash):
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get((messages[-1]["content"], params_hash))

    async def put(self, messages, params_hash, answer):
        if self.fail:
            raise ConnectionError("redis down")
        self.puts += 1
        self.store[(messages[-1]["content"], params_hash)] = answer


class DownRedis:
    async def incr(self, key):
        raise ConnectionError("redis down")


def make_deps(rate_limit: int = 60, **overrides) -> Deps:
    settings = Settings(
        _env_file=None, database_url="x", redis_url="x", rate_limit_per_minute=rate_limit
    )
    deps = {
        "settings": settings,
        "pool": FakePool(),
        "redis": fakeredis.FakeAsyncRedis(),
        "providers": FakeProviders(),
        "router": FakeRouter(),
        "cache": FakeCache(),
        "prices": load_prices(Path("config/prices.yaml")),
    }
    deps.update(overrides)
    return Deps(**deps)


def client(deps: Deps) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ASGITransport(create_app(deps)), base_url="http://test")


def body(**kw) -> dict:
    return {"model": "auto", "messages": [{"role": "user", "content": "hi"}], **kw}


def log_column(deps: Deps, row: int, name: str):
    return deps.pool.logs[row][LOG_COLUMNS.index(name)]


async def test_missing_key_401():
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json=body())
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


async def test_stream_rejected():
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json=body(stream=True), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "stream_not_supported"


async def test_unknown_model_rejected():
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json=body(model="gpt-4o"), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_found"


async def test_image_content_rejected():
    msg = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json=body(messages=[msg]), headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content"


async def test_malformed_body_400():
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json={"model": "auto"}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


async def test_forced_small_headers():
    deps = make_deps()
    async with client(deps) as c:
        r = await c.post("/v1/chat/completions", json=body(model="small"), headers=AUTH)
    assert r.status_code == 200
    assert r.headers["x-route"] == "small"
    Decimal(r.headers["x-notional-cost-usd"])
    assert len(deps.pool.logs) == 1
    assert log_column(deps, 0, "route") == "small"


async def test_small_bad_answer_escalates():
    deps = make_deps(providers=FakeProviders(small=""))
    async with client(deps) as c:
        r = await c.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert deps.providers.calls == ["small", "large"]
    assert r.headers["x-route"] == "large"
    assert log_column(deps, 0, "fallback") is True


async def test_small_down_escalates():
    deps = make_deps(providers=FakeProviders(small=ProviderError("small_unavailable")))
    async with client(deps) as c:
        r = await c.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 200
    assert log_column(deps, 0, "fallback") is True


async def test_all_down_503():
    providers = FakeProviders(large=ProviderError("all_tiers_unavailable"))
    async with client(make_deps(providers=providers)) as c:
        r = await c.post("/v1/chat/completions", json=body(model="large"), headers=AUTH)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_unavailable"


async def test_cache_hit_skips_providers():
    deps = make_deps()
    async with client(deps) as c:
        await c.post("/v1/chat/completions", json=body(temperature=0), headers=AUTH)
        r = await c.post("/v1/chat/completions", json=body(temperature=0), headers=AUTH)
    assert r.headers["x-route"] == "cache"
    assert len(deps.providers.calls) == 1


async def test_fallback_answer_not_cached():
    deps = make_deps(providers=FakeProviders(small=""))
    async with client(deps) as c:
        await c.post("/v1/chat/completions", json=body(temperature=0), headers=AUTH)
    assert deps.cache.puts == 0


async def test_redis_down_fails_open():
    deps = make_deps(redis=DownRedis(), cache=FakeCache(fail=True))
    async with client(deps) as c:
        r = await c.post("/v1/chat/completions", json=body(temperature=0), headers=AUTH)
    assert r.status_code == 200


async def test_rate_limit_429():
    async with client(make_deps(rate_limit=1)) as c:
        await c.post("/v1/chat/completions", json=body(), headers=AUTH)
        r = await c.post("/v1/chat/completions", json=body(), headers=AUTH)
    assert r.status_code == 429
    assert "retry-after" in r.headers


async def test_openai_sdk_compatible():
    app = create_app(make_deps())
    sdk = AsyncOpenAI(
        base_url="http://test/v1",
        api_key=KEY,
        http_client=httpx.AsyncClient(transport=ASGITransport(app)),
    )
    reply = await sdk.chat.completions.create(
        model="small", messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.choices[0].message.content == "fake answer"


@pytest.mark.parametrize("content", ["hi", [{"type": "text", "text": "hi"}]])
async def test_content_parts_accepted(content):
    msgs = [{"role": "user", "content": content}]
    async with client(make_deps()) as c:
        r = await c.post("/v1/chat/completions", json=body(messages=msgs), headers=AUTH)
    assert r.status_code == 200

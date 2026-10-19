import pytest
from redis.asyncio import Redis

from app.cache import SemanticCache
from app.features import Embedder

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


@pytest.fixture
async def cache(redis_url):
    redis = Redis.from_url(redis_url)
    await redis.flushdb()
    cache = SemanticCache(redis, Embedder(), threshold=0.95, ttl_s=60)
    await cache.ensure_index()
    yield cache
    await redis.aclose()


async def test_paraphrase_hits(cache):
    await cache.put(user("What is the capital of France?"), "p1", "Paris.")
    assert await cache.get(user("what's the capital of france"), "p1") == "Paris."


async def test_different_params_miss(cache):
    await cache.put(user("What is the capital of France?"), "p1", "Paris.")
    assert await cache.get(user("What is the capital of France?"), "p2") is None


async def test_unrelated_miss(cache):
    await cache.put(user("What is the capital of France?"), "p1", "Paris.")
    assert await cache.get(user("How do I bake bread?"), "p1") is None

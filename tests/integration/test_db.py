from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from app.config import load_prices
from app.db import (
    RequestLogEntry,
    create_api_key,
    create_pool,
    fetch_stats,
    insert_request_log,
    verify_api_key,
)

pytestmark = pytest.mark.integration
PRICES = load_prices(Path("config/prices.yaml"))


@pytest.fixture
async def pool(pg_url):
    pool = await create_pool(pg_url)
    await pool.execute("TRUNCATE request_log, api_keys")
    yield pool
    await pool.close()


async def test_key_roundtrip(pool):
    raw = await create_api_key(pool, "test")
    assert isinstance(await verify_api_key(pool, raw), UUID)
    assert await verify_api_key(pool, "gw_wrong") is None


async def test_raw_key_not_stored(pool):
    raw = await create_api_key(pool, "test")
    hashes = [r["key_hash"] for r in await pool.fetch("SELECT key_hash FROM api_keys")]
    assert raw not in hashes
    assert all(len(h) == 64 for h in hashes)


async def test_inactive_key_rejected(pool):
    raw = await create_api_key(pool, "test")
    await pool.execute("UPDATE api_keys SET active = false")
    assert await verify_api_key(pool, raw) is None


async def test_stats_aggregates(pool):
    key_id = await verify_api_key(pool, await create_api_key(pool, "test"))
    for route, latency in [("small", 100), ("small", 200), ("large", 300), ("cache", 5)]:
        await insert_request_log(
            pool,
            RequestLogEntry(
                api_key_id=key_id,
                route=route,
                model="m",
                router_score=0.5,
                fallback=False,
                cache_hit=route == "cache",
                prompt_tokens=1000,
                completion_tokens=1000,
                latency_ms=latency,
                notional_cost_usd=Decimal("0.001"),
                status=200,
                error=None,
            ),
        )
    stats = await fetch_stats(pool, timedelta(hours=24), PRICES)
    expected_p95 = await pool.fetchval(
        "SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x)"
        " FROM unnest(ARRAY[100, 200, 300, 5]) AS x"
    )
    assert stats["total"] == 4
    assert stats["by_route"] == {"small": 2, "large": 1, "cache": 1}
    assert stats["cache_hit_rate"] == 0.25
    assert stats["p95_latency_ms"] == pytest.approx(expected_p95)
    assert stats["always_large_cost_usd"] == pytest.approx(4 * (0.00059 + 0.00079))

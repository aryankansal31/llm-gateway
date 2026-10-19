"""Postgres access: API keys, the request log and the /stats aggregates."""

import hashlib
import secrets
from dataclasses import astuple, dataclass, fields
from datetime import timedelta
from decimal import Decimal
from uuid import UUID

import asyncpg

from app.config import TierPrice
from app.costs import notional_cost


async def create_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=1, max_size=10)


def _hash(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def create_api_key(pool: asyncpg.Pool, name: str) -> str:
    """Create a key and return it raw. Only its SHA-256 is stored; this is the only copy."""
    raw = "gw_" + secrets.token_urlsafe(32)
    await pool.execute("INSERT INTO api_keys (name, key_hash) VALUES ($1, $2)", name, _hash(raw))
    return raw


async def verify_api_key(pool: asyncpg.Pool, raw_key: str) -> UUID | None:
    return await pool.fetchval(
        "SELECT id FROM api_keys WHERE key_hash = $1 AND active", _hash(raw_key)
    )


@dataclass
class RequestLogEntry:
    api_key_id: UUID | None
    route: str
    model: str | None
    router_score: float | None
    fallback: bool
    cache_hit: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    notional_cost_usd: Decimal
    status: int
    error: str | None


_LOG_COLUMNS = [f.name for f in fields(RequestLogEntry)]
_INSERT_LOG = (
    f"INSERT INTO request_log ({', '.join(_LOG_COLUMNS)}) "
    f"VALUES ({', '.join(f'${i + 1}' for i in range(len(_LOG_COLUMNS)))})"
)


async def insert_request_log(pool: asyncpg.Pool, entry: RequestLogEntry) -> None:
    await pool.execute(_INSERT_LOG, *astuple(entry))


_STATS_SQL = """
SELECT
    count(*) AS total,
    coalesce(avg(cache_hit::int), 0) AS cache_hit_rate,
    coalesce(avg(fallback::int), 0) AS fallback_rate,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
    coalesce(sum(notional_cost_usd), 0) AS cost,
    coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
    coalesce(sum(completion_tokens), 0) AS completion_tokens
FROM request_log
WHERE ts >= now() - $1::interval
"""


async def fetch_stats(pool: asyncpg.Pool, window: timedelta, prices: dict[str, TierPrice]) -> dict:
    row = await pool.fetchrow(_STATS_SQL, window)
    by_route = await pool.fetch(
        "SELECT route, count(*) AS n FROM request_log WHERE ts >= now() - $1::interval"
        " GROUP BY route",
        window,
    )
    # Baseline: the same traffic, every request sent to the large tier.
    always_large = notional_cost("large", row["prompt_tokens"], row["completion_tokens"], prices)
    return {
        "total": row["total"],
        "by_route": {r["route"]: r["n"] for r in by_route},
        "cache_hit_rate": float(row["cache_hit_rate"]),
        "fallback_rate": float(row["fallback_rate"]),
        "p50_latency_ms": row["p50"],
        "p95_latency_ms": row["p95"],
        "notional_cost_usd": float(row["cost"]),
        "always_large_cost_usd": float(always_large),
    }

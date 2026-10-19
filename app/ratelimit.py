"""Fixed-window rate limit per API key, using Redis INCR + EXPIRE (spec 5.7)."""

import time
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: int


async def check_rate_limit(
    redis: Redis, key_id: str, limit: int, window_s: int = 60
) -> RateLimitResult:
    now = time.time()
    window = int(now) // window_s
    key = f"rl:{key_id}:{window}"
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_s)
    retry_after = max(1, (window + 1) * window_s - int(now))
    return RateLimitResult(allowed=count <= limit, retry_after=retry_after)

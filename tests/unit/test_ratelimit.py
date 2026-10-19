import fakeredis

from app.ratelimit import check_rate_limit


async def test_allows_up_to_limit():
    redis = fakeredis.FakeAsyncRedis()
    results = [await check_rate_limit(redis, "key_a", limit=3) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert 1 <= results[-1].retry_after <= 60


async def test_keys_are_independent():
    redis = fakeredis.FakeAsyncRedis()
    for _ in range(3):
        await check_rate_limit(redis, "key_a", limit=2)
    assert (await check_rate_limit(redis, "key_b", limit=2)).allowed


async def test_counter_expires():
    redis = fakeredis.FakeAsyncRedis()
    await check_rate_limit(redis, "key_a", limit=5)
    [key] = await redis.keys("rl:key_a:*")
    assert 0 < await redis.ttl(key) <= 60

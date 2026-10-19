from datetime import timedelta

import respx
from fastapi import FastAPI
from opentelemetry import trace

from app.telemetry import setup_tracing
from tests.unit.test_pipeline import client, make_deps


class DownPool:
    async def fetchval(self, *args):
        raise ConnectionError("postgres down")


async def test_models_list():
    async with client(make_deps()) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["data"]] == ["auto", "small", "large"]


async def test_stats_window_parsing(monkeypatch):
    seen: list[timedelta] = []

    async def fake_fetch_stats(pool, window, prices):
        seen.append(window)
        return {"total": 0}

    monkeypatch.setattr("app.main.db.fetch_stats", fake_fetch_stats)
    async with client(make_deps()) as c:
        ok = await c.get("/stats", params={"window": "7d"})
        bad = await c.get("/stats", params={"window": "abc"})
    assert ok.status_code == 200
    assert seen == [timedelta(days=7)]
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_window"


@respx.mock
async def test_health_reports_down():
    respx.get("http://localhost:11434/api/version").respond(200, json={"version": "x"})
    async with client(make_deps(pool=DownPool())) as c:
        r = await c.get("/health")
    assert r.status_code == 503
    assert r.json()["postgres"] == "down"
    assert r.json()["redis"] == "ok"


def test_tracing_noop_without_endpoint():
    before = trace.get_tracer_provider()
    setup_tracing(FastAPI(), None)
    assert trace.get_tracer_provider() is before

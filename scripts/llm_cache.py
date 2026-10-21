"""Synchronous model calls for the offline scripts, cached in SQLite (spec 5.3).

Every call is keyed by its model, messages and parameters, so labelling can be spread over
several days of free-tier quota and re-runs cost nothing.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

from app.config import Settings, TierConfig

MAX_RATE_LIMIT_RETRIES = 5
DEFAULT_RETRY_AFTER_S = 20


def build_clients(settings: Settings, tiers: dict[str, TierConfig]) -> dict[str, tuple]:
    """Map every tier to (client, model). Groq tiers are skipped when no key is set."""
    ollama = OpenAI(base_url=settings.ollama_base_url, api_key="ollama", max_retries=0)
    groq = (
        OpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key, max_retries=0)
        if settings.groq_api_key
        else None
    )
    clients = {}
    for name, tier in tiers.items():
        client = groq if tier.provider == "groq" else ollama
        if client is not None:
            clients[name] = (client, tier.model)
    return clients


class CachedLLM:
    def __init__(self, db_path: Path, clients: dict[str, tuple[OpenAI, str]]):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response_json TEXT)"
        )
        self.clients = clients

    def complete(
        self, tier: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024
    ) -> dict:
        client, model = self.clients[tier]
        key = hashlib.sha256(
            json.dumps([model, messages, temperature, max_tokens], sort_keys=True).encode()
        ).hexdigest()
        row = self.db.execute("SELECT response_json FROM calls WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            start = time.perf_counter()
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                break
            except RateLimitError as e:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                wait = float(e.response.headers.get("retry-after", DEFAULT_RETRY_AFTER_S))
                print(f"[{tier}] rate limited; sleeping {wait:.0f}s")
                time.sleep(wait)

        result = {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
        self.db.execute("INSERT INTO calls VALUES (?, ?)", (key, json.dumps(result)))
        self.db.commit()
        return result

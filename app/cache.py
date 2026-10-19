"""Semantic cache: answers a repeated question by meaning, using Redis vector search (spec 5.4)."""

import asyncio
import re
import uuid

import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.features import EMBED_DIM, Embedder, last_user_text
from app.schemas import ChatCompletionRequest

INDEX = "gwcache"
PREFIX = "gwcache:"


def cacheable(req: ChatCompletionRequest) -> bool:
    """Single-turn, near-deterministic requests only, so stale context never leaks into an answer."""
    roles = [m.role for m in req.messages]
    return (
        roles.count("user") == 1
        and roles.count("assistant") == 0
        and roles.count("system") <= 1
        and req.temperature <= 0.3
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class SemanticCache:
    def __init__(self, redis: Redis, embedder: Embedder, threshold: float, ttl_s: int):
        self.redis = redis
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s

    async def ensure_index(self) -> None:
        try:
            await self.redis.ft(INDEX).info()
            return
        except ResponseError:
            pass
        await self.redis.ft(INDEX).create_index(
            [
                VectorField(
                    "vec",
                    "HNSW",
                    {"TYPE": "FLOAT32", "DIM": EMBED_DIM, "DISTANCE_METRIC": "COSINE"},
                ),
                TagField("ctx"),
            ],
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
        )

    async def _vector(self, messages: list[dict]) -> bytes:
        text = _normalise(last_user_text(messages))
        # Embedding is CPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(self.embedder.embed, [text])
        return vectors[0].astype(np.float32).tobytes()

    async def get(self, messages: list[dict], params_hash: str) -> str | None:
        query = (
            Query(f"(@ctx:{{{params_hash}}})=>[KNN 1 @vec $vec AS dist]")
            .return_fields("answer", "dist")
            .dialect(2)
        )
        result = await self.redis.ft(INDEX).search(
            query, query_params={"vec": await self._vector(messages)}
        )
        if not result.docs:
            return None
        doc = result.docs[0]
        if 1 - float(doc.dist) < self.threshold:
            return None
        answer = doc.answer
        return answer.decode() if isinstance(answer, bytes) else answer

    async def put(self, messages: list[dict], params_hash: str, answer: str) -> None:
        key = f"{PREFIX}{uuid.uuid4().hex}"
        mapping = {"vec": await self._vector(messages), "ctx": params_hash, "answer": answer}
        await self.redis.hset(key, mapping=mapping)
        await self.redis.expire(key, self.ttl_s)

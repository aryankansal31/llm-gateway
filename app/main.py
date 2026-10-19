"""FastAPI app: the OpenAI-compatible endpoint and the request pipeline (spec 4 and 6)."""

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import db
from app.cache import cacheable
from app.config import Settings, TierPrice, get_settings, load_prices, load_tiers
from app.costs import notional_cost
from app.features import estimate_tokens
from app.providers import ProviderError, ProviderResult
from app.quality import check_answer
from app.ratelimit import check_rate_limit
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
    error_response,
)

log = logging.getLogger(__name__)

MODELS = ("auto", "small", "large")
# Redis failures in the cache or the rate limiter never fail a request (Review Focus 4).
REDIS_ERRORS = (RedisError, OSError)


@dataclass
class Deps:
    settings: Settings
    pool: Any
    redis: Any
    providers: Any
    router: Any
    cache: Any
    prices: dict[str, TierPrice]


async def build_deps(settings: Settings) -> Deps:
    from redis.asyncio import Redis

    from app.cache import SemanticCache
    from app.features import Embedder
    from app.providers import Providers
    from app.router import Router

    embedder = Embedder()
    redis = Redis.from_url(settings.redis_url)
    cache = SemanticCache(
        redis, embedder, settings.cache_similarity_threshold, settings.cache_ttl_seconds
    )
    try:
        await cache.ensure_index()
    except REDIS_ERRORS as e:
        log.warning("could not create the cache index: %s", e)
    return Deps(
        settings=settings,
        pool=await db.create_pool(settings.database_url),
        redis=redis,
        providers=Providers(settings, load_tiers(settings.models_config_path)),
        router=Router(settings.router_model_path, embedder, settings.small_max_prompt_tokens),
        cache=cache,
        prices=load_prices(settings.prices_config_path),
    )


def params_hash(req: ChatCompletionRequest, key_id) -> str:
    """Cache context: an entry only matches the same API key, system prompt and parameters."""
    system = next((m.text() for m in req.messages if m.role == "system"), None)
    params = {
        "api_key_id": str(key_id),
        "system_prompt": system,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "response_format": req.response_format,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def create_app(deps: Deps | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if deps is None:
            app.state.deps = await build_deps(get_settings())
        yield
        if deps is None:
            built: Deps = app.state.deps
            await built.pool.close()
            await built.redis.aclose()
            await built.providers.aclose()

    app = FastAPI(title="Cost-aware LLM Gateway", version="0.1.0", lifespan=lifespan)
    if deps is not None:
        app.state.deps = deps

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(400, str(exc.errors()), "invalid_request_error")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, str(exc.detail), "invalid_request_error")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await handle_completion(request, request.app.state.deps)

    return app


async def handle_completion(request: Request, deps: Deps) -> JSONResponse:
    # 1. Auth
    auth = request.headers.get("authorization", "")
    raw_key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
    try:
        key_id = await db.verify_api_key(deps.pool, raw_key) if raw_key else None
    except Exception as e:  # Postgres down: a clean 503, never a raw 500 (Review Focus 3)
        log.error("cannot verify API key: %s", e)
        msg = "Key store unavailable."
        return error_response(503, msg, "server_error", "auth_unavailable")
    if key_id is None:
        return error_response(401, "Invalid API key.", "authentication_error", "invalid_api_key")

    # 2. Rate limit (fails open)
    try:
        limit = await check_rate_limit(deps.redis, str(key_id), deps.settings.rate_limit_per_minute)
        if not limit.allowed:
            return error_response(
                429,
                "Rate limit exceeded.",
                "rate_limit_error",
                "rate_limit_exceeded",
                headers={"Retry-After": str(limit.retry_after)},
            )
    except REDIS_ERRORS as e:
        log.warning("rate limiter unavailable, allowing request: %s", e)

    # 3. Validate
    try:
        req = ChatCompletionRequest.model_validate(await request.json())
    except (ValidationError, ValueError) as e:
        return error_response(400, f"Invalid request: {e}", "invalid_request_error")
    if req.stream:
        msg = "Streaming is not supported."
        return error_response(400, msg, "invalid_request_error", "stream_not_supported")
    if req.model not in MODELS:
        msg = f"Unknown model {req.model!r}. Use one of: {', '.join(MODELS)}."
        return error_response(400, msg, "invalid_request_error", "model_not_found")
    try:
        messages = [{"role": m.role, "content": m.text()} for m in req.messages]
    except ValueError:
        msg = "Only text content parts are supported."
        return error_response(400, msg, "invalid_request_error", "unsupported_content")

    start = time.perf_counter()
    use_cache = cacheable(req)
    p_hash = params_hash(req, key_id) if use_cache else ""

    # 4. Semantic cache (fails open)
    if use_cache:
        try:
            cached = await deps.cache.get(messages, p_hash)
        except REDIS_ERRORS as e:
            log.warning("cache unavailable, skipping lookup: %s", e)
            cached = None
        if cached is not None:
            prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))
            result = ProviderResult(
                cached, "stop", prompt_tokens, estimate_tokens(cached), "cache", "cache"
            )
            return await finish(deps, key_id, result, None, False, start)

    # 5. Route
    decision = await asyncio.to_thread(deps.router.decide, req.model, messages)

    # 6-7. Dispatch, with quality-check fallback from small to large
    json_mode = (req.response_format or {}).get("type") == "json_object"
    args = (messages, req.temperature, req.max_tokens, json_mode)
    fallback = False
    try:
        result = None
        if decision.tier == "small":
            try:
                result = await deps.providers.complete("small", *args)
                reason = check_answer(result.text, result.finish_reason, json_mode)
                if reason:
                    log.info("small answer failed the quality check (%s); escalating", reason)
                    result, fallback = None, True
            except ProviderError as e:
                log.warning("small tier unavailable (%s); escalating", e)
                fallback = True
        if result is None:
            result = await deps.providers.complete("large", *args)
    except ProviderError as e:
        await log_request(
            deps,
            db.RequestLogEntry(
                api_key_id=key_id,
                route=decision.tier,
                model=None,
                router_score=decision.score,
                fallback=fallback,
                cache_hit=False,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                notional_cost_usd=Decimal(0),
                status=503,
                error=str(e),
            ),
        )
        msg = "No model is available to serve this request."
        return error_response(503, msg, "server_error", "upstream_unavailable")

    # 8. Store in the cache: only clean, first-try answers (Review Focus 5)
    if use_cache and not fallback:
        try:
            await deps.cache.put(messages, p_hash, result.text)
        except REDIS_ERRORS as e:
            log.warning("cache unavailable, not storing: %s", e)

    return await finish(deps, key_id, result, decision.score, fallback, start)


async def finish(
    deps: Deps,
    key_id,
    result: ProviderResult,
    score: float | None,
    fallback: bool,
    start: float,
) -> JSONResponse:
    """9. Log the request and build the response with the x- headers."""
    latency_ms = int((time.perf_counter() - start) * 1000)
    cost = notional_cost(result.tier, result.prompt_tokens, result.completion_tokens, deps.prices)
    await log_request(
        deps,
        db.RequestLogEntry(
            api_key_id=key_id,
            route=result.tier,
            model=result.model,
            router_score=score,
            fallback=fallback,
            cache_hit=result.tier == "cache",
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=latency_ms,
            notional_cost_usd=cost,
            status=200,
            error=None,
        ),
    )
    response = ChatCompletionResponse(
        model=result.model,
        choices=[
            Choice(message=ResponseMessage(content=result.text), finish_reason=result.finish_reason)
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
    headers = {
        "x-route": result.tier,
        "x-router-score": "none" if score is None else f"{score:.4f}",
        "x-notional-cost-usd": str(cost),
        "x-latency-ms": str(latency_ms),
    }
    return JSONResponse(response.model_dump(), headers=headers)


async def log_request(deps: Deps, entry: db.RequestLogEntry) -> None:
    try:
        await db.insert_request_log(deps.pool, entry)
    except Exception as e:  # logging must never cost the caller their answer
        log.warning("could not write request log: %s", e)

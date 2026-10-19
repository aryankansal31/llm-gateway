# Cost-aware LLM Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OpenAI-compatible FastAPI gateway that routes each chat request to a small local model or a large free-tier model using a trained Scikit-learn classifier, with a Redis semantic cache, Postgres request logging, notional cost tracking, tracing, and a reproducible benchmark.

**Architecture:** An async FastAPI app (`app/`) handles auth, rate limiting, caching, routing, dispatch, quality-check fallback and logging. Offline scripts (`scripts/`) build a labelled dataset from public prompts, train the router, and run the benchmark; every model call they make is cached in SQLite. Feature extraction lives in one module shared by training and serving.

**Tech Stack:** Python 3.12, uv, FastAPI, Pydantic v2, pydantic-settings, openai SDK (`AsyncOpenAI` and `OpenAI`), asyncpg, redis-py (asyncio + search), fastembed (`BAAI/bge-small-en-v1.5`), NumPy, Pandas, Scikit-learn, joblib, Matplotlib, Hugging Face `datasets`, OpenTelemetry + OpenInference + Arize Phoenix, pytest, pytest-asyncio, respx, fakeredis, ruff, Docker Compose, GitHub Actions.

**Spec:** `docs/architecture.md`

## Global Constraints

- Free only: no paid service and no credit card. Models: Ollama (local) and Groq free tier. Everything else open source, self-hosted.
- No LiteLLM or any other gateway or router library.
- Python 3.12; dependencies managed with `uv`; lint with `ruff check` and `ruff format --check`.
- Model names live only in `config/models.yaml`: `small` = `llama3.2:3b` (ollama), `large` = `llama-3.3-70b-versatile` (groq), `large_offline` = `llama3.1:8b` (ollama), `judge` = a Groq model from a different family than `large`.
- Ollama runs on the host. From containers it is `http://host.docker.internal:11434/v1`; from the host it is `http://localhost:11434/v1`.
- Groq base URL: `https://api.groq.com/openai/v1`. Missing `GROQ_API_KEY` means `large` is always served by `large_offline`.
- `model` values accepted: `auto` (routing), `small`, `large`. Anything else → `400`. `stream: true` → `400`.
- Errors use the OpenAI envelope: `{"error": {"message": str, "type": str, "code": str | null}}`.
- Response headers on every completion: `x-route`, `x-router-score`, `x-notional-cost-usd`, `x-latency-ms`.
- `route` values in logs and headers: `cache`, `small`, `large`, `large_offline`.
- Cache: single-turn requests only, `temperature <= 0.3`, cosine similarity `>= 0.95`, TTL `86400` s, 384-dim vectors.
- Rate limit: fixed window, default `60` requests per `60` s per key; `429` with `Retry-After`.
- API keys: shown once, stored as SHA-256 hex in Postgres.
- Router threshold: lowest `t` whose routed quality on the validation split is `>= 0.95` of always-large.
- Dataset: 600 prompts, split 70/15/15 stratified by `source` and `small_ok`; seed `42` everywhere.
- No real model calls in CI. Tests that need Postgres or Redis are marked `@pytest.mark.integration` and read `TEST_DATABASE_URL` / `TEST_REDIS_URL`; they are skipped when those are unset.
- Every number shown in the README comes from a committed file under `results/` produced by a script in this repo.

## Review Focus

1. **OpenAI content-part messages** (`"content": [{"type": "text", "text": "hi"}]`, which some SDK clients send): text parts are joined with `"\n"`; any non-text part returns `400`. The test is in Task 9.
2. **A prompt longer than the small model's context:** Ollama silently truncates past `num_ctx`, so a prompt estimated at more than `small_max_prompt_tokens` (default `3000`) must go to `large` whatever the router score. The test is in Task 8.
3. **Ollama not running or the model not pulled:** a small-tier connection error escalates to `large` with `fallback = true`. If every tier fails, the gateway returns `503` with the error envelope, never a raw 500 traceback. Tests are in Tasks 4 and 9.
4. **Redis down:** cache and rate limiting fail open with a logged warning, and the request is still served. The test is in Task 9.
5. **Cache poisoning:** a response is cached only when its status is 200, it passed the quality check, and it was not a fallback from a failed small answer. Errors and escalated answers are never stored. The test is in Task 9.

---

## File Structure

```
app/
  config.py      Settings (env) + YAML loaders for tiers and prices
  schemas.py     OpenAI-compatible request/response models, error envelope helper
  costs.py       notional cost calculation
  quality.py     small-answer quality checks
  providers.py   AsyncOpenAI clients per tier, offline fallback, ProviderResult
  db.py          asyncpg pool, api keys, request log, stats query
  ratelimit.py   Redis fixed-window limiter
  features.py    embeddings + handcrafted features (shared with training)
  router.py      load joblib artifact, decide tier
  cache.py       Redis vector semantic cache
  telemetry.py   OpenTelemetry → Phoenix setup
  main.py        create_app(), endpoints, request pipeline
scripts/
  __init__.py
  llm_cache.py   SQLite-cached synchronous model calls for offline scripts
  create_key.py  CLI: create an API key
  label.py       build data/labels.parquet
  train_router.py  train, calibrate, choose threshold, write models/router-v1.*
  benchmark.py   produce results/benchmark.md and cost_quality_curve.png
config/models.yaml  config/prices.yaml
sql/init.sql
models/  results/  data/ (gitignored)
tests/unit/  tests/integration/  tests/conftest.py
Dockerfile  docker-compose.yml  pyproject.toml  .env.example  .github/workflows/ci.yml
```

---

### Task 1: Project scaffold, config and CI

**Files:**
- Create: `pyproject.toml`, `.env.example`, `config/models.yaml`, `config/prices.yaml`, `app/__init__.py`, `app/config.py`, `tests/conftest.py`, `tests/unit/test_config.py`, `.github/workflows/ci.yml`, `docker-compose.yml`, `sql/init.sql`
- Modify: `.gitignore` (add `data/`, `!data/.gitkeep`)

**Interfaces:**
- Produces:
  - `class TierConfig(BaseModel): provider: Literal["ollama", "groq"]; model: str`
  - `class TierPrice(BaseModel): usd_per_1m_input: Decimal; usd_per_1m_output: Decimal`
  - `class Settings(BaseSettings)` with fields: `database_url: str`, `redis_url: str`, `groq_api_key: str | None = None`, `ollama_base_url: str = "http://localhost:11434/v1"`, `groq_base_url: str = "https://api.groq.com/openai/v1"`, `models_config_path: Path = Path("config/models.yaml")`, `prices_config_path: Path = Path("config/prices.yaml")`, `router_model_path: Path = Path("models/router-v1.joblib")`, `rate_limit_per_minute: int = 60`, `cache_similarity_threshold: float = 0.95`, `cache_ttl_seconds: int = 86400`, `small_max_prompt_tokens: int = 3000`, `phoenix_endpoint: str | None = None`
  - `load_tiers(path: Path) -> dict[str, TierConfig]` (keys: `small`, `large`, `large_offline`, `judge`)
  - `load_prices(path: Path) -> dict[str, TierPrice]` (keys: `small`, `large`, `large_offline`)
  - `get_settings() -> Settings` (cached with `functools.lru_cache`)

- [ ] **Step 1: Create `pyproject.toml`** with the dependencies from the Tech Stack line (runtime) and `pytest`, `pytest-asyncio`, `respx`, `fakeredis`, `ruff` (dev group). Configure `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` and marker `integration`. Configure ruff with `line-length = 100`, `target-version = "py312"`. Run `uv sync`. Expected: a lockfile `uv.lock` is created.

- [ ] **Step 2: Write `config/models.yaml` and `config/prices.yaml`**

`prices.yaml` holds notional reference prices, dated, with source:

```yaml
as_of: "2026-10-19"   # re-check on the day you run the benchmark and update this date
source: "Groq on-demand list prices for comparable hosted models"
tiers:
  small:         { usd_per_1m_input: 0.06, usd_per_1m_output: 0.06 }
  large:         { usd_per_1m_input: 0.59, usd_per_1m_output: 0.79 }
  large_offline: { usd_per_1m_input: 0.05, usd_per_1m_output: 0.08 }
```

- [ ] **Step 3: Write failing tests** in `tests/unit/test_config.py`:
  - `test_load_tiers_reads_all_four_tiers`: `set(load_tiers(Path("config/models.yaml"))) == {"small", "large", "large_offline", "judge"}` and `tiers["small"].model == "llama3.2:3b"`.
  - `test_judge_is_different_model_than_large`: `tiers["judge"].model != tiers["large"].model`.
  - `test_load_prices_returns_decimals`: `prices["large"].usd_per_1m_output == Decimal("0.79")`.
  - `test_settings_defaults`: with `DATABASE_URL` and `REDIS_URL` set via `monkeypatch`, `Settings().rate_limit_per_minute == 60` and `cache_similarity_threshold == 0.95`.

- [ ] **Step 4: Run** `uv run pytest tests/unit/test_config.py -v`. Expected: FAIL, `ModuleNotFoundError: app.config`.

- [ ] **Step 5: Implement `app/config.py`** with the interfaces above. `Settings` reads `.env`.

- [ ] **Step 6: Run** `uv run pytest tests/unit/test_config.py -v`. Expected: 4 passed.

- [ ] **Step 7: Write `sql/init.sql`** with the `api_keys` and `request_log` tables exactly as in spec §5.8, plus `CREATE EXTENSION IF NOT EXISTS pgcrypto;` for `gen_random_uuid()` defaults and an index on `request_log(ts)`.

- [ ] **Step 8: Write `docker-compose.yml`** with services `postgres` (`postgres:16`, mounts `sql/init.sql` into `/docker-entrypoint-initdb.d/`), `redis` (`redis:8`), `phoenix` (`arizephoenix/phoenix`, pinned to a specific version tag, UI on port `6006`, OTLP on `4317`). The `gateway` service is added in Task 11. Run `docker compose up -d` then `docker compose exec postgres psql -U postgres -c "\dt"`. Expected: `api_keys` and `request_log` listed.

- [ ] **Step 9: Write `tests/conftest.py`** with fixtures `pg_url` and `redis_url` that `pytest.skip` when `TEST_DATABASE_URL` / `TEST_REDIS_URL` are unset.

- [ ] **Step 10: Write `.github/workflows/ci.yml`**: on push and pull_request; services `postgres:16` and `redis:8` with health checks; steps: checkout, `astral-sh/setup-uv`, `uv sync`, `psql -f sql/init.sql`, `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -v` with `TEST_DATABASE_URL` and `TEST_REDIS_URL` set.

- [ ] **Step 11: Write `.env.example`** listing every `Settings` field with safe placeholder values and the comment `# Get a free key (no card) at https://console.groq.com`.

- [ ] **Step 12: Commit and push**, then check the run with `gh run watch`. Expected: CI green.

```bash
git add -A && git commit -m "chore: scaffold project, config loaders, compose stack and CI"
git push
```

---

### Task 2: Notional cost

**Files:**
- Create: `app/costs.py`, `tests/unit/test_costs.py`

**Interfaces:**
- Consumes: `TierPrice`, `load_prices` (Task 1)
- Produces: `notional_cost(tier: str, prompt_tokens: int, completion_tokens: int, prices: dict[str, TierPrice]) -> Decimal`. The `cache` tier costs `Decimal("0")`. Result quantized to 8 decimal places.

- [ ] **Step 1: Write failing tests:**
  - `test_large_cost`: `notional_cost("large", 1_000_000, 1_000_000, prices) == Decimal("1.38000000")`
  - `test_small_cost_small_numbers`: `notional_cost("small", 500, 200, prices) == Decimal("0.00004200")`
  - `test_cache_is_free`: `notional_cost("cache", 10_000, 10_000, prices) == Decimal("0E-8")` (compare with `== 0`)
  - `test_unknown_tier_raises`: `pytest.raises(KeyError)` for `"gpt-9"`
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_costs.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement `notional_cost`** in `app/costs.py`.
- [ ] **Step 4: Run** the same command. Expected: 4 passed.
- [ ] **Step 5: Commit** `feat: notional cost calculation`.

---

### Task 3: Quality checks

**Files:**
- Create: `app/quality.py`, `tests/unit/test_quality.py`

**Interfaces:**
- Produces: `check_answer(text: str, finish_reason: str | None, wants_json: bool) -> str | None`. Returns `None` when the answer passes, otherwise a reason code: `"empty"`, `"truncated"`, `"refusal"` or `"invalid_json"`.

- [ ] **Step 1: Write failing tests** (one per reason, plus a pass):
  - `check_answer("   ", "stop", False) == "empty"`
  - `check_answer("Paris is", "length", False) == "truncated"`
  - `check_answer("I can't help with that request.", "stop", False) == "refusal"`
  - `check_answer("As an AI language model, I cannot browse.", "stop", False) == "refusal"`
  - `check_answer("{not json", "stop", True) == "invalid_json"`
  - `check_answer('{"a": 1}', "stop", True) is None`
  - `check_answer("The capital of France is Paris.", "stop", False) is None`
  - `test_refusal_only_checked_at_start`: `check_answer("Here is how. Note: I can't guarantee prices.", "stop", False) is None`
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_quality.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `check_answer`.** Refusal detection: a case-insensitive regex matched only against the first 120 characters, with patterns `^(i can'?t|i cannot|i'm unable|i am unable|sorry, but i)` and `as an ai`.
- [ ] **Step 4: Run.** Expected: 8 passed.
- [ ] **Step 5: Commit** `feat: small-answer quality checks`.

---

### Task 4: Providers with offline fallback

**Files:**
- Create: `app/providers.py`, `tests/unit/test_providers.py`

**Interfaces:**
- Consumes: `Settings`, `TierConfig` (Task 1)
- Produces:
  - `@dataclass class ProviderResult: text: str; finish_reason: str | None; prompt_tokens: int; completion_tokens: int; model: str; tier: str`
  - `class ProviderError(Exception)` (raised when no tier can serve)
  - `class Providers: def __init__(self, settings: Settings, tiers: dict[str, TierConfig]); async def complete(self, tier: Literal["small", "large"], messages: list[dict], temperature: float, max_tokens: int | None, json_mode: bool) -> ProviderResult`

Behaviour:
- `tier="large"`: call Groq. On `RateLimitError`, `APIStatusError` with status ≥ 500, `APIConnectionError`, or no API key, call `large_offline` and return `tier="large_offline"`.
- `tier="small"`: call Ollama. On `APIConnectionError` or `NotFoundError` (model not pulled), raise `ProviderError("small_unavailable")`. Escalation is the pipeline's job (Task 9).
- If `large_offline` also fails, raise `ProviderError("all_tiers_unavailable")`.
- `json_mode=True` sends `response_format={"type": "json_object"}`.

- [ ] **Step 1: Write failing tests** using `respx` to mock `http://localhost:11434/v1/chat/completions` and `https://api.groq.com/openai/v1/chat/completions`:
  - `test_large_uses_groq`: Groq returns 200 → `result.tier == "large"` and token counts come from `usage`.
  - `test_large_falls_back_on_429`: Groq 429, Ollama 200 → `result.tier == "large_offline"` and `result.model == "llama3.1:8b"`.
  - `test_large_falls_back_without_key`: `groq_api_key=None` → Groq route never called (`respx` route `.called is False`), `tier == "large_offline"`.
  - `test_small_connection_error_raises`: Ollama raises `httpx.ConnectError` → `ProviderError` with `str(e) == "small_unavailable"`.
  - `test_everything_down_raises`: Groq 503 and Ollama `ConnectError` → `ProviderError("all_tiers_unavailable")`.
  - `test_json_mode_sends_response_format`: request body sent to Groq has `response_format == {"type": "json_object"}`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_providers.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `Providers`** with one `AsyncOpenAI` per base URL (Ollama's `api_key="ollama"`), `max_retries=0` so the fallback logic is ours, and `timeout=60`.
- [ ] **Step 4: Run.** Expected: 6 passed.
- [ ] **Step 5: Commit** `feat: model providers with offline fallback`.

---

### Task 5: Database access, API keys and key CLI

**Files:**
- Create: `app/db.py`, `scripts/__init__.py`, `scripts/create_key.py`, `tests/integration/test_db.py`

**Interfaces:**
- Produces:
  - `async def create_pool(database_url: str) -> asyncpg.Pool`
  - `async def create_api_key(pool, name: str) -> str` returns the raw key `"gw_" + secrets.token_urlsafe(32)`; stores only its SHA-256 hex.
  - `async def verify_api_key(pool, raw_key: str) -> UUID | None` (`None` if unknown or inactive)
  - `@dataclass class RequestLogEntry` with the `request_log` columns except `id` and `ts`
  - `async def insert_request_log(pool, entry: RequestLogEntry) -> None`
  - `async def fetch_stats(pool, window: timedelta, prices: dict[str, TierPrice]) -> dict` returns keys `total`, `by_route` (dict route → count), `cache_hit_rate`, `fallback_rate`, `p50_latency_ms`, `p95_latency_ms`, `notional_cost_usd`, `always_large_cost_usd`. The last is computed in Python as summed tokens × the `large` price.
  - CLI: `uv run python -m scripts.create_key --name <name>` prints the key once.

- [ ] **Step 1: Write failing integration tests** (marked `integration`, use `pg_url`; truncate tables in a fixture):
  - `test_key_roundtrip`: `verify_api_key(pool, raw)` returns a UUID; `verify_api_key(pool, "gw_wrong")` is `None`.
  - `test_raw_key_not_stored`: `SELECT key_hash FROM api_keys` contains no row equal to `raw`, and `len(key_hash) == 64`.
  - `test_inactive_key_rejected`: after `UPDATE api_keys SET active=false`, `verify_api_key` returns `None`.
  - `test_stats_aggregates`: insert 4 log rows (routes `small`, `small`, `large`, `cache`; latencies 100, 200, 300, 5) → `by_route == {"small": 2, "large": 1, "cache": 1}`, `cache_hit_rate == 0.25`, `p95_latency_ms` equals `percentile_cont(0.95)` of those latencies, computed in SQL.
- [ ] **Step 2: Run** `docker compose up -d postgres` and then `TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres uv run pytest tests/integration/test_db.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `app/db.py` and `scripts/create_key.py`.** Use `percentile_cont` in SQL for the latencies.
- [ ] **Step 4: Run.** Expected: 4 passed.
- [ ] **Step 5: Commit** `feat: postgres access, api keys, stats query`.

---

### Task 6: Rate limiter

**Files:**
- Create: `app/ratelimit.py`, `tests/unit/test_ratelimit.py`

**Interfaces:**
- Produces: `@dataclass class RateLimitResult: allowed: bool; retry_after: int` and `async def check_rate_limit(redis: Redis, key_id: str, limit: int, window_s: int = 60) -> RateLimitResult`. Redis key: `rl:{key_id}:{int(time.time()) // window_s}`. Use `INCR`, and set `EXPIRE window_s` when the count is 1.

- [ ] **Step 1: Write failing tests** with `fakeredis.aioredis.FakeRedis`:
  - `test_allows_up_to_limit`: 3 calls with `limit=3` are all allowed; the 4th has `allowed is False` and `1 <= retry_after <= 60`.
  - `test_keys_are_independent`: exhausting `key_a` does not block `key_b`.
  - `test_counter_expires`: after the first call, the TTL of the window key is in `(0, 60]`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_ratelimit.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 3 passed.
- [ ] **Step 5: Commit** `feat: fixed-window rate limiter`.

---

### Task 7: Feature builder

**Files:**
- Create: `app/features.py`, `tests/unit/test_features.py`

**Interfaces:**
- Produces:
  - `EMBED_DIM = 384`
  - `HANDCRAFTED_NAMES: list[str]` = `["n_tokens_est", "n_questions", "has_code", "digit_ratio", "has_math_symbols", "has_reasoning_words", "n_turns"]`
  - `estimate_tokens(text: str) -> int` = `ceil(len(text) / 4)`
  - `last_user_text(messages: list[dict]) -> str`
  - `handcrafted(messages: list[dict]) -> np.ndarray` (shape `(7,)`, float32, order as `HANDCRAFTED_NAMES`)
  - `class Embedder: def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"); def embed(self, texts: list[str]) -> np.ndarray` (shape `(n, 384)`, L2-normalised)
  - `build_features(messages_batch: list[list[dict]], embedder: Embedder) -> np.ndarray` (shape `(n, 391)`: embedding columns first, then handcrafted)

Handcrafted definitions: `has_code` = text contains a triple backtick, or `def ` / `class ` / `function ` / `;\n`. `has_math_symbols` = any of `=+*/^√∑∫` or `\d+\s*[x×]\s*\d+`. `has_reasoning_words` = case-insensitive match of `prove|step by step|explain why|compare|derive|analy[sz]e`. `n_turns` = count of `user` messages.

- [ ] **Step 1: Write failing tests:**
  - `test_handcrafted_shape_and_order`: for `[{"role":"user","content":"What is 12 x 7? Explain why."}]` → shape `(7,)`, `n_questions == 1`, `has_math_symbols == 1`, `has_reasoning_words == 1`, `n_turns == 1`.
  - `test_code_detection`: a message containing a fenced code block → `has_code == 1`.
  - `test_last_user_text_skips_assistant`: returns the content of the last `user` message.
  - `test_build_features_shape` with a fake `Embedder` subclass returning `np.ones((n, 384))/sqrt(384)`: shape `(2, 391)`.
  - `test_real_embedder_normalised` (marked `slow`, skipped in CI via `-m "not slow"`): `np.allclose(np.linalg.norm(Embedder().embed(["hi"]), axis=1), 1.0, atol=1e-3)`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_features.py -v -m "not slow"`. Expected: FAIL.
- [ ] **Step 3: Implement** using `fastembed.TextEmbedding`. Embed the `last_user_text`. Register the `slow` marker in `pyproject.toml` and add `-m "not slow"` to the CI pytest command.
- [ ] **Step 4: Run** without the marker filter once locally. Expected: 5 passed (the first run downloads the ONNX model).
- [ ] **Step 5: Commit** `feat: shared feature builder for router`.

---

### Task 8: Router (inference)

**Files:**
- Create: `app/router.py`, `tests/unit/test_router.py`

**Interfaces:**
- Consumes: `build_features`, `estimate_tokens`, `Embedder` (Task 7)
- Produces:
  - `@dataclass class RouteDecision: tier: Literal["small", "large"]; score: float | None; reason: str` (reason ∈ `"model"`, `"forced"`, `"too_long"`, `"no_model"`)
  - `class Router: def __init__(self, model_path: Path, embedder: Embedder, small_max_prompt_tokens: int); def decide(self, requested_model: str, messages: list[dict]) -> RouteDecision`
  - The artifact format written by Task 13: a joblib dict `{"model": <fitted classifier with predict_proba>, "threshold": float, "feature_version": 1}`.

Decision order:
1. `requested_model` is `small` or `large` → that tier, `reason="forced"`, `score=None`.
2. Estimated tokens of all messages > `small_max_prompt_tokens` → `large`, `"too_long"`.
3. Artifact missing → `large`, `"no_model"`.
4. `p = predict_proba(features)[0, 1]`; `small` if `p >= threshold`, else `large`; `reason="model"`, `score=p`.

- [ ] **Step 1: Write failing tests** with a fake embedder and a tiny `LogisticRegression` fitted on random 391-dim data, saved to `tmp_path / "r.joblib"`:
  - `test_forced_tier`: `decide("small", msgs).tier == "small"` and `reason == "forced"`.
  - `test_too_long_goes_large` (Review Focus 2): a message of 13,000 characters with `small_max_prompt_tokens=3000` → `tier == "large"` and `reason == "too_long"`, even with `threshold=0.0`.
  - `test_missing_artifact_goes_large`: nonexistent path → `tier == "large"` and `reason == "no_model"`.
  - `test_threshold_zero_routes_small`: artifact with `threshold=0.0` → `tier == "small"`, `0 <= score <= 1`.
  - `test_threshold_one_routes_large`: artifact with `threshold=1.01` → `tier == "large"`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_router.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `Router`.** Load the artifact once in `__init__`; assert `feature_version == 1`.
- [ ] **Step 4: Run.** Expected: 5 passed.
- [ ] **Step 5: Commit** `feat: router inference with forced, too-long and missing-model rules`.

---

### Task 9: Semantic cache and the completion endpoint

**Files:**
- Create: `app/schemas.py`, `app/cache.py`, `app/main.py`, `tests/unit/test_schemas.py`, `tests/unit/test_pipeline.py`, `tests/integration/test_cache.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces:
  - `schemas.py`:
    - `ChatCompletionRequest` (`model: str`, `messages: list[ChatMessage]`, `temperature: float = 1.0`, `max_tokens: int | None = None`, `response_format: dict | None = None`, `stream: bool = False`)
    - `ChatMessage` (`role: Literal["system","user","assistant"]`, `content: str | list[dict]`) with `def text(self) -> str`
    - `ChatCompletionResponse` matching the OpenAI `chat.completion` object (`id`, `object`, `created`, `model`, `choices[{index, message, finish_reason}]`, `usage{prompt_tokens, completion_tokens, total_tokens}`)
    - `def error_response(status: int, message: str, type_: str, code: str | None = None) -> JSONResponse`
  - `cache.py`:
    - `class SemanticCache: def __init__(self, redis, embedder, threshold: float, ttl_s: int)`
    - `async def ensure_index(self) -> None` creates index `gwcache` on prefix `gwcache:` with field `vec` (HNSW, FLOAT32, DIM 384, COSINE) and tag `ctx`
    - `async def get(self, messages, params_hash: str) -> str | None`
    - `async def put(self, messages, params_hash: str, answer: str) -> None`
    - `def cacheable(req: ChatCompletionRequest) -> bool` (single user turn, at most one system message, `temperature <= 0.3`)
  - `main.py`:
    - `@dataclass class Deps` (`settings`, `pool`, `redis`, `providers`, `router`, `cache`, `prices`)
    - `def create_app(deps: Deps | None = None) -> FastAPI` (when `deps` is `None`, the lifespan builds real deps from `get_settings()`)
    - Endpoints: `POST /v1/chat/completions`, `GET /v1/models`, `GET /health`

Pipeline for `POST /v1/chat/completions`, in order:
1. Bearer auth via `verify_api_key` → `401` `invalid_api_key`.
2. Rate limit → `429` with `Retry-After`. **On a Redis error, allow the request and log a warning.**
3. Validate: `stream` → `400` `stream_not_supported`; model not in {`auto`, `small`, `large`} → `400` `model_not_found`; non-text content part → `400` `unsupported_content`.
4. If cacheable: `cache.get` → on hit, return with `route="cache"`. **On a Redis error, skip the cache.**
5. `router.decide`.
6. Dispatch. If small: call it; on `ProviderError("small_unavailable")` or a non-`None` `check_answer`, call large and set `fallback=True`.
7. On `ProviderError("all_tiers_unavailable")` → `503` `upstream_unavailable`.
8. If cacheable **and** status 200 **and** `fallback is False`: `cache.put`.
9. `insert_request_log` (on a DB error, log a warning and still return the answer). Set the four `x-` headers. Latency is measured with `time.perf_counter` around steps 4–8.

`params_hash` = SHA-256 of the JSON `{system_prompt, temperature, max_tokens, response_format}` (keys sorted).

- [ ] **Step 1: Write failing schema tests** in `test_schemas.py`:
  - `test_content_parts_joined` (Review Focus 1): `ChatMessage(role="user", content=[{"type":"text","text":"a"},{"type":"text","text":"b"}]).text() == "a\nb"`.
  - `test_image_part_rejected`: `.text()` on content with `{"type":"image_url",...}` raises `ValueError("unsupported_content")`.
- [ ] **Step 2: Write failing pipeline tests** in `test_pipeline.py` using `create_app(Deps(...))` with fakes (a `FakeProviders` that records calls, a fake router, `fakeredis` for the rate limit, a `FakeCache` dict-backed, a fake pool with an in-memory key and log list) and `httpx.AsyncClient(transport=ASGITransport(app))`:
  - `test_missing_key_401`: no header → 401, body `error.code == "invalid_api_key"`.
  - `test_stream_rejected`: `stream: true` → 400 `stream_not_supported`.
  - `test_unknown_model_rejected`: `model: "gpt-4o"` → 400 `model_not_found`.
  - `test_forced_small_headers`: `model: "small"` → 200, `x-route == "small"`, `x-notional-cost-usd` parses as a Decimal, one log row with `route == "small"`.
  - `test_small_bad_answer_escalates`: small returns `""` → large is called, `x-route == "large"`, log `fallback is True`.
  - `test_small_down_escalates` (Review Focus 3): small raises `ProviderError("small_unavailable")` → 200 from large, `fallback is True`.
  - `test_all_down_503` (Review Focus 3): large raises `ProviderError("all_tiers_unavailable")` → 503 `upstream_unavailable`, error envelope present.
  - `test_cache_hit_skips_providers`: the second identical request with `temperature: 0` → `x-route == "cache"`, providers called once in total.
  - `test_fallback_answer_not_cached` (Review Focus 5): the small answer fails the check → `FakeCache.put` is never called.
  - `test_redis_down_fails_open` (Review Focus 4): rate-limit redis and cache both raise `ConnectionError` → 200 is still returned.
  - `test_rate_limit_429`: with `rate_limit_per_minute=1`, the second request gets 429 with a `Retry-After` header.
  - `test_openai_sdk_compatible`: `AsyncOpenAI(base_url="http://test/v1", api_key=key, http_client=httpx.AsyncClient(transport=ASGITransport(app)))` → `chat.completions.create(model="small", ...)` returns an object with `.choices[0].message.content == "fake answer"`.
- [ ] **Step 3: Write a failing integration test** in `tests/integration/test_cache.py` (real Redis 8 via `redis_url`, real `Embedder`, marked `integration` and `slow`):
  - `test_paraphrase_hits`: `put` "What is the capital of France?" → `get` "what's the capital of france" returns the answer.
  - `test_different_params_miss`: same text with a different `params_hash` → `None`.
  - `test_unrelated_miss`: "How do I bake bread?" → `None`.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_schemas.py tests/unit/test_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 5: Implement `schemas.py`, `cache.py` and `main.py`.** Use `redis.commands.search` (`VectorField("vec", "HNSW", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"})`) and a KNN 1 query filtered by `@ctx:{params_hash}`. A hit requires `1 - distance >= threshold`. Store with `EXPIRE ttl_s`.
- [ ] **Step 6: Run** the unit tests. Expected: 14 passed. Then run the cache integration tests with Redis up. Expected: 3 passed.
- [ ] **Step 7: Commit** `feat: completion endpoint with routing, fallback, semantic cache, logging`.

---

### Task 10: Stats, models list and tracing

**Files:**
- Create: `app/telemetry.py`, `tests/unit/test_endpoints_misc.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `fetch_stats` (Task 5), `Deps`, `create_app` (Task 9)
- Produces:
  - `GET /stats?window=24h` (accepts `<int>h` or `<int>d`; anything else → `400` `invalid_window`) returns the `fetch_stats` dict as JSON.
  - `GET /v1/models` returns `{"object": "list", "data": [{"id": "auto", "object": "model"}, {"id": "small", ...}, {"id": "large", ...}]}`.
  - `GET /health` returns 200 `{"redis": "ok", "postgres": "ok", "ollama": "ok"}`, or 503 with `"down"` for any failing check.
  - `setup_tracing(app: FastAPI, endpoint: str | None) -> None` is a no-op when `endpoint` is `None`. Otherwise it registers an OTLP exporter to Phoenix and `OpenAIInstrumentor().instrument()`; the pipeline creates spans named `cache.lookup`, `router.decide`, `model.call` and `fallback`.

- [ ] **Step 1: Write failing tests:**
  - `test_models_list`: the ids are exactly `["auto", "small", "large"]`.
  - `test_stats_window_parsing`: `window=7d` → 200; `window=abc` → 400 `invalid_window`.
  - `test_health_reports_down`: a fake pool that raises → 503 and `postgres == "down"`.
  - `test_tracing_noop_without_endpoint`: `setup_tracing(app, None)` does not raise and registers no span processor.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_endpoints_misc.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the endpoints and `telemetry.py`.
- [ ] **Step 4: Run.** Expected: 4 passed.
- [ ] **Step 5: Manual check:** `docker compose up -d`, then `uv run uvicorn app.main:create_app --factory`, create a key, send one `model: "small"` request with curl, and open `http://localhost:6006`. Expected: one trace with the four spans.
- [ ] **Step 6: Commit** `feat: stats, models list, health, tracing`.

---

### Task 11: Container image and compose service

**Files:**
- Create: `Dockerfile`, `.dockerignore`
- Modify: `docker-compose.yml`

**Interfaces:**
- Produces: a `gateway` compose service on port `8000`, with `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`, `extra_hosts: ["host.docker.internal:host-gateway"]`, and dependencies on healthy `postgres` and `redis`.

- [ ] **Step 1: Write the `Dockerfile`:** `python:3.12-slim`, install `uv`, `uv sync --frozen --no-dev`, pre-download the fastembed model at build time (`python -c "from app.features import Embedder; Embedder()"`), and `CMD uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000`.
- [ ] **Step 2: Run** `docker compose up -d --build`, then `curl -s localhost:8000/health`. Expected: `{"redis":"ok","postgres":"ok","ollama":"ok"}` (Ollama running on the host with `llama3.2:3b` pulled).
- [ ] **Step 3: Smoke-test with the OpenAI SDK** from the host. Run `uv run python -c` with `OpenAI(base_url="http://localhost:8000/v1", api_key=KEY).chat.completions.create(model="auto", messages=[{"role":"user","content":"Say hi"}])`. Expected: a reply. `/stats` then shows `total == 1`.
- [ ] **Step 4: Commit** `feat: dockerised gateway service`.

---

### Task 12: Cached offline LLM calls and the labelling script

**Files:**
- Create: `scripts/llm_cache.py`, `scripts/label.py`, `tests/unit/test_llm_cache.py`, `tests/unit/test_label_rules.py`

**Interfaces:**
- Consumes: `load_tiers`, `Settings` (Task 1)
- Produces:
  - `scripts/llm_cache.py`:
    - `class CachedLLM: def __init__(self, db_path: Path, clients: dict[str, tuple[OpenAI, str]])`, where the dict maps tier name to (client, model)
    - `def complete(self, tier: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> dict` returns `{"text", "prompt_tokens", "completion_tokens", "latency_ms"}`. The SQLite table `calls(key TEXT PRIMARY KEY, response_json TEXT)` is keyed by SHA-256 of `json.dumps([model, messages, temperature, max_tokens], sort_keys=True)`. On a 429 it sleeps for the `Retry-After` value (default 20 s) and retries, up to 5 times.
  - `scripts/label.py`:
    - `SOURCES` = `{"dolly": 200, "gsm8k": 120, "mmlu": 120, "humaneval": 80, "mtbench": 80}`, loaded from `databricks/databricks-dolly-15k` (train), `openai/gsm8k` config `main` (test), `cais/mmlu` config `all` (test, formatted as the question plus lettered choices), `openai/openai_humaneval` (test, the `prompt` field), and `HuggingFaceH4/mt_bench_prompts` (first turn). The last is a builder substitution for the spec's `lmsys/mt_bench_human_judgments`: same 80 MT-Bench questions, cleaner format.
    - `def small_ok(verdict_small_first: str, verdict_large_first: str) -> int`: a small win scores 1, a tie 0.5, a loss 0, averaged over both orders; returns `1` if the mean is `>= 0.5`, else `0`.
    - `JUDGE_PROMPT`: asks for the single token `A`, `B` or `tie`. Any other output is re-asked once, then treated as `tie`.
    - CLI: `uv run python -m scripts.label --limit 600`. It is resumable, because every call is cached. It writes `data/labels.parquet` with columns `id, source, prompt, small_answer, large_answer, verdict_small_first, verdict_large_first, small_ok, small_prompt_tokens, small_completion_tokens, large_prompt_tokens, large_completion_tokens`.

- [ ] **Step 1: Write failing tests:**
  - `test_cache_hit_avoids_second_call`: a fake `OpenAI` client counting calls; two identical `complete` calls → 1 underlying call.
  - `test_cache_key_includes_model`: the same messages with a different tier model → 2 calls.
  - `test_small_ok_rules`: `small_ok("A","B") == 1` (small wins in both orders, since small is A first and B second); `small_ok("B","A") == 0`; `small_ok("tie","tie") == 1`; `small_ok("B","tie") == 0` (mean 0.25); `small_ok("A","A") == 1` (mean 0.5: one win and one loss means position bias, so it counts as a tie).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_llm_cache.py tests/unit/test_label_rules.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement both scripts.** Sampling uses `random.Random(42)`.
- [ ] **Step 4: Run.** Expected: 3 passed.
- [ ] **Step 5: Pilot run:** `uv run python -m scripts.label --limit 20`. Expected: `data/labels.parquet` with 20 rows and both labels present. If one class is under 25% across the full 600 later, oversample `gsm8k` and `humaneval` (spec §12) and note it in the model card.
- [ ] **Step 6: Start the full run in the background** (`--limit 600`). It may take more than one day of free-tier quota; re-running resumes from the cache.
- [ ] **Step 7: Commit** `feat: cached offline LLM calls and labelling pipeline`.

---

### Task 13: Router training and model card

**Files:**
- Create: `scripts/train_router.py`, `tests/unit/test_train_router.py`
- Output (committed): `models/router-v1.joblib`, `models/router-v1.json`, `results/splits.json`

**Interfaces:**
- Consumes: `build_features`, `Embedder` (Task 7); `data/labels.parquet` (Task 12)
- Produces:
  - `def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` (70/15/15, stratified on `source + "_" + small_ok`)
  - `def routed_quality(p: np.ndarray, small_ok: np.ndarray, t: float) -> float` = mean of `(1 if p_i < t else small_ok_i)`
  - `def choose_threshold(p: np.ndarray, small_ok: np.ndarray, target: float = 0.95) -> float`: the lowest `t` in `np.round(np.arange(0, 1.01, 0.01), 2)` with `routed_quality >= target`; returns `1.01` if none qualifies
  - The artifact is the dict described in Task 8. `results/splits.json` = `{"train": [ids], "val": [ids], "test": [ids]}`.
  - The model card JSON has keys: `trained_at`, `dataset_sha256` (of `labels.parquet`), `n_train/n_val/n_test`, `label_balance`, `candidates` (per model: `val_roc_auc`), `chosen`, `threshold`, `val_routed_quality`, `val_share_small`, `test_roc_auc`, `test_precision`, `test_recall` (at the threshold), `feature_version: 1`.

- [ ] **Step 1: Write failing tests:**
  - `test_split_sizes_and_disjoint`: 600 synthetic rows → sizes `(420, 90, 90)` and no id overlap.
  - `test_routed_quality_all_large`: `routed_quality(p, ok, t=1.01) == 1.0`.
  - `test_routed_quality_all_small`: `routed_quality(p, ok, t=0.0) == ok.mean()`.
  - `test_choose_threshold_lowest_meeting_target`: `p=[0.9,0.8,0.2,0.1]`, `ok=[1,1,0,0]` → `0.21` (routes the two confident cases to small at full quality, and is the lowest threshold doing so).
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_train_router.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** Candidates: `LogisticRegression(class_weight="balanced", max_iter=2000)` (in a pipeline with `StandardScaler`) and `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)`. Pick by validation ROC-AUC, refit inside `CalibratedClassifierCV(method="sigmoid", cv=5)` on train, then choose the threshold on validation. Test metrics are computed once, at the end.
- [ ] **Step 4: Run** the tests. Expected: 4 passed.
- [ ] **Step 5: Train for real** (after Task 12 finishes): `uv run python -m scripts.train_router`. Expected: the three output files exist and `test_roc_auc` is printed. Restart the gateway, send `model: "auto"`, and expect `x-router-score` to be a number.
- [ ] **Step 6: Commit** `feat: router training, calibration, threshold selection, model card`.

---

### Task 14: Benchmark

**Files:**
- Create: `scripts/benchmark.py`, `tests/unit/test_benchmark.py`
- Output (committed): `results/benchmark.md`, `results/cost_quality_curve.png`, `results/judge_audit.md` (template)

**Interfaces:**
- Consumes: `results/splits.json`, `data/labels.parquet`, `models/router-v1.*`, the running gateway (Task 11), `CachedLLM` (Task 12), `notional_cost` (Task 2)
- Produces:
  - `def summarize(latencies_ms: list[float]) -> dict` → `{"p50": float, "p95": float}` (`numpy.percentile`, linear)
  - `def quality_table(test_df, p_test, threshold) -> dict[str, float]` for keys `always_small`, `always_large` (=1.0), `routed`, using the Task 13 `routed_quality` definition
  - CLI: `uv run python -m scripts.benchmark --gateway http://localhost:8000 --key $KEY`

Steps the script performs:
1. **Quality** comes from the existing judge labels for the test ids (no new judge calls).
2. **Notional cost per 1,000 requests** for each setup comes from the labelled token counts and `prices.yaml`. For routed, each prompt uses the small or large counts according to the router decision.
3. **Latency:** send every test prompt through the gateway three times, with `model` set to `small`, `large` and `auto`, at `temperature: 0.3` and with a unique system-message nonce per run so the cache never hits. Record client-side latency and report p50 and p95.
4. **Cache experiment:** generate one paraphrase per test prompt with `CachedLLM` (`large` tier, prompt `"Paraphrase this request without changing its meaning. Output only the paraphrase:\n\n{prompt}"`). Send the original, then the paraphrase, both at `temperature: 0`. Report the hit rate on paraphrases and the mean latency of hits versus misses.
5. **Curve:** for `t` in `0.00..1.00`, plot notional cost against routed quality on the test split, mark the chosen threshold, and save the PNG.
6. **Write `results/benchmark.md`:** the table (setup, quality, notional cost per 1K, p50, p95, share small), the cache results, the price table `as_of` date, the dataset hash, and the exact command used.
7. **Write `results/judge_audit.md`:** 30 random test items with both answers and the judge verdicts, plus an empty `human_verdict` column for Aryan to fill in.

- [ ] **Step 1: Write failing tests:**
  - `test_summarize_percentiles`: `summarize([100]*19 + [1000])["p95"] == pytest.approx(145.0)`.
  - `test_quality_table_keys`: returns exactly `{"always_small", "always_large", "routed"}` and `always_large == 1.0`.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_benchmark.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: 2 passed.
- [ ] **Step 5: Run the real benchmark** with the stack up. Expected: three files in `results/`, and a table with no empty cells.
- [ ] **Step 6: Commit** `feat: benchmark with quality, notional cost, latency and cache results`.

---

### Task 15: README results and release

**Files:**
- Modify: `README.md`

- [ ] **Step 1:** Replace every `TBD` in the README benchmark table with values copied from `results/benchmark.md`. Add the curve image, a "Router" section quoting `test_roc_auc`, precision and recall from `models/router-v1.json`, a CI badge, a demo GIF (record with ScreenToGif, which is free), and a Quick start: copy `.env.example`, `ollama pull llama3.2:3b llama3.1:8b`, `docker compose up -d --build`, `uv run python -m scripts.create_key --name demo`, then a curl example.
- [ ] **Step 2: Tick the README roadmap** items that are done.
- [ ] **Step 3: Verify:** `grep -c TBD README.md` prints `0`. Every number in the README appears in a file under `results/` or `models/`.
- [ ] **Step 4: Commit and tag**

```bash
git add README.md results/ && git commit -m "docs: publish benchmark results"
git tag v0.1.0 && git push --follow-tags
```

Expected: CI green on the tag commit.

-- Schema for the gateway. Runs once, on the first start of the Postgres container.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS api_keys (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    key_hash    text NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    active      boolean NOT NULL DEFAULT true
);

CREATE TABLE IF NOT EXISTS request_log (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                 timestamptz NOT NULL DEFAULT now(),
    api_key_id         uuid REFERENCES api_keys(id),
    route              text NOT NULL,   -- 'cache' | 'small' | 'large' | 'large_offline'
    model              text,
    router_score       real,
    fallback           boolean NOT NULL DEFAULT false,
    cache_hit          boolean NOT NULL DEFAULT false,
    prompt_tokens      int,
    completion_tokens  int,
    latency_ms         int,
    notional_cost_usd  numeric(12, 8),
    status             int NOT NULL,
    error              text
);

CREATE INDEX IF NOT EXISTS request_log_ts_idx ON request_log (ts);

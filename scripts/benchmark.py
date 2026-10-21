"""Benchmark (spec 7): uv run python -m scripts.benchmark --gateway http://localhost:8000 --key KEY

Quality and notional cost come from the judge labels and token counts of the held-out test
split. Latency and the cache experiment are measured live through the running gateway.
Writes results/benchmark.md, results/cost_quality_curve.png and results/judge_audit.md.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml

from app.config import get_settings, load_prices, load_tiers
from app.costs import notional_cost
from app.features import Embedder
from scripts.llm_cache import CachedLLM, build_clients
from scripts.train_router import ARTIFACT, LABELS, SPLITS, features, routed_quality

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
CACHE = Path("data/cache.sqlite")
PARAPHRASE_PROMPT = (
    "Paraphrase this request without changing its meaning. Output only the paraphrase:\n\n{prompt}"
)


def summarize(latencies_ms: list[float]) -> dict:
    return {
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
    }


def quality_table(test_df: pd.DataFrame, p_test: np.ndarray, threshold: float) -> dict[str, float]:
    ok = test_df["small_ok"].to_numpy()
    return {
        "always_small": routed_quality(p_test, ok, 0.0),
        "always_large": 1.0,
        "routed": routed_quality(p_test, ok, threshold),
    }


def cost_per_1k(test_df: pd.DataFrame, use_small: np.ndarray, prices) -> Decimal:
    total = Decimal(0)
    for row, small in zip(test_df.itertuples(), use_small, strict=True):
        tier = "small" if small else "large"
        total += notional_cost(
            tier,
            getattr(row, f"{tier}_prompt_tokens"),
            getattr(row, f"{tier}_completion_tokens"),
            prices,
        )
    return (total / len(test_df) * 1000).quantize(Decimal("0.0001"))


def send(http: httpx.Client, model: str, messages: list[dict], temperature: float):
    start = time.perf_counter()
    r = http.post(
        "/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": temperature},
    )
    latency = (time.perf_counter() - start) * 1000
    r.raise_for_status()
    return latency, r.headers.get("x-route")


def measure_latency(http: httpx.Client, prompts: list[str]) -> dict:
    """Each prompt once per setup, with a unique system nonce so the cache never answers."""
    out = {}
    for model in ("small", "large", "auto"):
        latencies, routes = [], []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": f"benchmark nonce {uuid.uuid4().hex}"},
                {"role": "user", "content": prompt},
            ]
            latency, route = send(http, model, messages, 0.3)
            latencies.append(latency)
            routes.append(route)
        out[model] = {**summarize(latencies), "routes": routes}
        print(f"latency {model}: {out[model]['p50']:.0f} / {out[model]['p95']:.0f} ms")
    return out


def cache_experiment(http: httpx.Client, llm: CachedLLM, prompts: list[str]) -> dict:
    hits, hit_latency, miss_latency = 0, [], []
    for prompt in prompts:
        paraphrase = llm.complete(
            "large", [{"role": "user", "content": PARAPHRASE_PROMPT.format(prompt=prompt)}]
        )["text"].strip()
        latency, _ = send(http, "auto", [{"role": "user", "content": prompt}], 0.0)
        miss_latency.append(latency)
        latency, route = send(http, "auto", [{"role": "user", "content": paraphrase}], 0.0)
        if route == "cache":
            hits += 1
            hit_latency.append(latency)
        else:
            miss_latency.append(latency)
    return {
        "paraphrase_hit_rate": hits / len(prompts),
        "mean_hit_ms": statistics.mean(hit_latency) if hit_latency else None,
        "mean_miss_ms": statistics.mean(miss_latency),
    }


def plot_curve(test_df, p_test, threshold, prices, path: Path) -> None:
    ok = test_df["small_ok"].to_numpy()
    ts = np.round(np.arange(0, 1.01, 0.01), 2)
    costs = [float(cost_per_1k(test_df, p_test >= t, prices)) for t in ts]
    quality = [routed_quality(p_test, ok, t) for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(costs, quality, marker=".", linewidth=1)
    chosen = int(np.argmin(np.abs(ts - threshold))) if threshold <= 1 else len(ts) - 1
    ax.scatter([costs[chosen]], [quality[chosen]], color="red", zorder=3, label=f"t = {threshold}")
    ax.set_xlabel("Notional cost per 1,000 requests (USD)")
    ax.set_ylabel("Quality relative to always-large")
    ax.set_title("Router threshold: cost versus quality (test split)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_audit(test_df: pd.DataFrame, path: Path) -> None:
    sample = test_df.sample(n=min(30, len(test_df)), random_state=42)
    lines = [
        "# Judge audit",
        "",
        "30 random test items. Read both answers and fill in `human_verdict` with small, large or "
        "tie, then compare with the judge's label.",
        "",
    ]
    for row in sample.itertuples():
        lines += [
            f"## {row.id} ({row.source})",
            "",
            f"**Judge:** small-first `{row.verdict_small_first}`, "
            f"large-first `{row.verdict_large_first}` → small_ok = {row.small_ok}",
            "",
            "**human_verdict:** ",
            "",
            "<details><summary>Prompt and answers</summary>",
            "",
            "**Prompt**",
            "",
            "```text",
            row.prompt,
            "```",
            "",
            "**Small answer**",
            "",
            "```text",
            row.small_answer,
            "```",
            "",
            "**Large answer**",
            "",
            "```text",
            row.large_answer,
            "```",
            "",
            "</details>",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def ms(value) -> str:
    return "n/a" if value is None else f"{value:,.0f} ms"


def main(gateway: str, key: str) -> None:
    settings = get_settings()
    prices = load_prices(settings.prices_config_path)
    as_of = yaml.safe_load(settings.prices_config_path.read_text(encoding="utf-8"))["as_of"]
    labels = pd.read_parquet(LABELS)
    test_ids = json.loads(SPLITS.read_text(encoding="utf-8"))["test"]
    test_df = labels.set_index("id").loc[test_ids].reset_index()

    artifact = joblib.load(ARTIFACT)
    threshold = artifact["threshold"]
    p_test = artifact["model"].predict_proba(features(test_df, Embedder()))[:, 1]
    use_small = p_test >= threshold

    quality = quality_table(test_df, p_test, threshold)
    costs = {
        "always_small": cost_per_1k(test_df, np.ones(len(test_df), bool), prices),
        "always_large": cost_per_1k(test_df, np.zeros(len(test_df), bool), prices),
        "routed": cost_per_1k(test_df, use_small, prices),
    }

    prompts = test_df["prompt"].tolist()
    llm = CachedLLM(CACHE, build_clients(settings, load_tiers(settings.models_config_path)))
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=gateway, headers=headers, timeout=180) as http:
        latency = measure_latency(http, prompts)
        cache = cache_experiment(http, llm, prompts)

    RESULTS.mkdir(exist_ok=True)
    plot_curve(test_df, p_test, threshold, prices, RESULTS / "cost_quality_curve.png")
    write_audit(test_df, RESULTS / "judge_audit.md")

    share_small = {"always_small": 1.0, "always_large": 0.0, "routed": float(use_small.mean())}
    rows = [
        ("Always large model", "always_large", "large"),
        ("Always small model", "always_small", "small"),
        ("**Routed**", "routed", "auto"),
    ]
    saving = 1 - costs["routed"] / costs["always_large"] if costs["always_large"] else Decimal(0)
    dataset_sha = hashlib.sha256(LABELS.read_bytes()).hexdigest()
    md = [
        "# Benchmark results",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} on the held-out test split "
        f"({len(test_df)} prompts).",
        "",
        "| Setup | Quality score | Notional cost per 1K requests | p50 latency | p95 latency "
        "| Share sent to small |",
        "|---|---|---|---|---|---|",
    ]
    for label, key_, model in rows:
        md.append(
            f"| {label} | {quality[key_]:.3f} | ${costs[key_]} | {ms(latency[model]['p50'])} "
            f"| {ms(latency[model]['p95'])} | {share_small[key_]:.0%} |"
        )
    md += [
        "",
        f"Routing at threshold **{threshold}** cuts notional cost by **{saving:.0%}** against "
        f"always-large while keeping **{quality['routed']:.1%}** of its quality.",
        "",
        "Quality: share of prompts where the served answer is at least as good as the large "
        "model's, according to the pairwise judge (both answer orders). Always-large is 1.0 by "
        "definition.",
        "",
        "## Semantic cache",
        "",
        f"- Paraphrase hit rate: **{cache['paraphrase_hit_rate']:.0%}**",
        f"- Mean latency of a hit: {ms(cache['mean_hit_ms'])}; of a miss: "
        f"{ms(cache['mean_miss_ms'])}",
        "",
        "## Provenance",
        "",
        f"- Prices: `config/prices.yaml`, as of {as_of}. All models are free; cost is notional.",
        f"- Dataset `data/labels.parquet`, sha256 `{dataset_sha}`",
        f"- Command: `uv run python -m scripts.benchmark --gateway {gateway} --key <key>`",
        "",
    ]
    (RESULTS / "benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    random.seed(42)
    main(args.gateway, args.key)

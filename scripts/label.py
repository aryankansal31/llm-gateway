"""Build the router's labelled dataset (spec 5.3): uv run python -m scripts.label --limit 600

For each prompt, the small and large models both answer, and a judge from a different model
family compares the answers in both orders. small_ok = 1 when small is at least as good.
Every model call is cached, so the script resumes where it stopped.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from openai import APIError

from app.config import get_settings, load_tiers
from scripts.llm_cache import CachedLLM, build_clients

SEED = 42
SOURCES = {"dolly": 200, "gsm8k": 120, "mmlu": 120, "humaneval": 80, "mtbench": 80}
OUT = Path("data/labels.parquet")
CACHE = Path("data/cache.sqlite")

JUDGE_PROMPT = """You are comparing two answers to the same request.

Request:
{prompt}

Answer A:
{a}

Answer B:
{b}

Which answer is better: more correct, more helpful and more complete? If they are about equally
good, say tie. Reply with exactly one token: A, B or tie."""


def _dolly(row: dict) -> str:
    context = (row.get("context") or "").strip()
    return row["instruction"] + (f"\n\n{context}" if context else "")


def _mmlu(row: dict) -> str:
    choices = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", row["choices"], strict=False))
    return f"{row['question']}\n\n{choices}\n\nAnswer with the correct letter and a short reason."


LOADERS = {
    "dolly": (("databricks/databricks-dolly-15k",), {"split": "train"}, _dolly),
    "gsm8k": (("openai/gsm8k", "main"), {"split": "test"}, lambda r: r["question"]),
    "mmlu": (("cais/mmlu", "all"), {"split": "test"}, _mmlu),
    "humaneval": (
        ("openai/openai_humaneval",),
        {"split": "test"},
        lambda r: f"Complete this Python function:\n\n{r['prompt']}",
    ),
    # Builder substitution for lmsys/mt_bench_human_judgments: the same 80 MT-Bench questions.
    "mtbench": (
        ("HuggingFaceH4/mt_bench_prompts",),
        {"split": "train"},
        lambda r: r["prompt"][0],
    ),
}


def sample_prompts(limit: int) -> list[dict]:
    """Stratified sample. A smaller `limit` takes a prefix of each source's full sample,
    so a pilot run's calls are reused (from the cache) by the full run."""
    rng = random.Random(SEED)
    total = sum(SOURCES.values())
    rows = []
    for source, n in SOURCES.items():
        args, kwargs, fmt = LOADERS[source]
        ds = load_dataset(*args, **kwargs)
        picked = rng.sample(range(len(ds)), min(n, len(ds)))
        k = max(1, round(n * min(limit, total) / total))
        for idx in picked[:k]:
            rows.append({"id": f"{source}-{idx}", "source": source, "prompt": fmt(ds[idx])})
    return rows[:limit]


def parse_verdict(text: str) -> str | None:
    token = text.strip().strip(".").strip().lower()
    return {"a": "A", "b": "B", "tie": "tie"}.get(token)


def judge(llm: CachedLLM, prompt: str, a: str, b: str) -> str:
    content = JUDGE_PROMPT.format(prompt=prompt, a=a, b=b)
    messages = [{"role": "user", "content": content}]
    verdict = parse_verdict(llm.complete("judge", messages)["text"])
    if verdict is None:  # re-ask once with a firmer instruction, then call it a tie
        retry = messages + [{"role": "user", "content": "Reply with only A, B or tie."}]
        verdict = parse_verdict(llm.complete("judge", retry)["text"])
    return verdict or "tie"


def small_ok(verdict_small_first: str, verdict_large_first: str) -> int:
    """1 when small is at least as good as large, averaged over both answer orders."""
    first = {"A": 1.0, "tie": 0.5, "B": 0.0}[verdict_small_first]  # small is A
    second = {"B": 1.0, "tie": 0.5, "A": 0.0}[verdict_large_first]  # small is B
    return int((first + second) / 2 >= 0.5)


def label_one(llm: CachedLLM, item: dict) -> dict:
    messages = [{"role": "user", "content": item["prompt"]}]
    small = llm.complete("small", messages)
    large = llm.complete("large", messages)
    v1 = judge(llm, item["prompt"], small["text"], large["text"])
    v2 = judge(llm, item["prompt"], large["text"], small["text"])
    return {
        **item,
        "small_answer": small["text"],
        "large_answer": large["text"],
        "verdict_small_first": v1,
        "verdict_large_first": v2,
        "small_ok": small_ok(v1, v2),
        "small_prompt_tokens": small["prompt_tokens"],
        "small_completion_tokens": small["completion_tokens"],
        "large_prompt_tokens": large["prompt_tokens"],
        "large_completion_tokens": large["completion_tokens"],
    }


def main(limit: int) -> int:
    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
        return 1
    llm = CachedLLM(CACHE, build_clients(settings, load_tiers(settings.models_config_path)))
    items = sample_prompts(limit)
    rows, stopped = [], None
    for i, item in enumerate(items, 1):
        try:
            rows.append(label_one(llm, item))
        except APIError as e:
            stopped = e
            break
        if i % 10 == 0:
            print(f"labelled {i}/{len(items)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT}")
    if len(df):
        print(df.groupby("source")["small_ok"].agg(["mean", "count"]))
        print(f"overall small_ok share: {df['small_ok'].mean():.2f}")
    if stopped:
        print(f"stopped early ({stopped}); re-run the same command later to resume")
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=600)
    sys.exit(main(parser.parse_args().limit))

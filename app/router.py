"""Decide which tier serves a request, using the trained classifier from scripts/train_router.py."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib

from app.features import Embedder, build_features, estimate_tokens

log = logging.getLogger(__name__)

FEATURE_VERSION = 1


@dataclass
class RouteDecision:
    tier: Literal["small", "large"]
    score: float | None
    reason: str  # "model" | "forced" | "too_long" | "no_model"


class Router:
    def __init__(self, model_path: Path, embedder: Embedder, small_max_prompt_tokens: int):
        self.embedder = embedder
        self.small_max_prompt_tokens = small_max_prompt_tokens
        self.artifact = None
        if model_path.exists():
            # Trusted input: the artifact is produced by scripts/train_router.py in this repo.
            self.artifact = joblib.load(model_path)
            assert self.artifact["feature_version"] == FEATURE_VERSION
        else:
            log.warning("router artifact %s not found; every auto request goes large", model_path)

    def decide(self, requested_model: str, messages: list[dict]) -> RouteDecision:
        if requested_model in ("small", "large"):
            return RouteDecision(requested_model, None, "forced")
        # Ollama silently truncates past its context window, so long prompts never go small.
        prompt_tokens = estimate_tokens("\n".join(m["content"] for m in messages))
        if prompt_tokens > self.small_max_prompt_tokens:
            return RouteDecision("large", None, "too_long")
        if self.artifact is None:
            return RouteDecision("large", None, "no_model")
        features = build_features([messages], self.embedder)
        p = float(self.artifact["model"].predict_proba(features)[0, 1])
        tier = "small" if p >= self.artifact["threshold"] else "large"
        return RouteDecision(tier, p, "model")

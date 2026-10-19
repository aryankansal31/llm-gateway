import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.router import Router
from tests.unit.test_features import FakeEmbedder

MSGS = [{"role": "user", "content": "What is the capital of France?"}]


def artifact(tmp_path, threshold):
    rng = np.random.default_rng(42)
    model = LogisticRegression().fit(rng.normal(size=(40, 391)), np.arange(40) % 2)
    path = tmp_path / "r.joblib"
    joblib.dump({"model": model, "threshold": threshold, "feature_version": 1}, path)
    return path


def router(path, max_tokens=3000):
    return Router(path, FakeEmbedder(), small_max_prompt_tokens=max_tokens)


def test_forced_tier(tmp_path):
    decision = router(artifact(tmp_path, 0.5)).decide("small", MSGS)
    assert decision.tier == "small"
    assert decision.reason == "forced"
    assert decision.score is None


def test_too_long_goes_large(tmp_path):
    msgs = [{"role": "user", "content": "a" * 13_000}]
    decision = router(artifact(tmp_path, 0.0)).decide("auto", msgs)
    assert decision.tier == "large"
    assert decision.reason == "too_long"


def test_missing_artifact_goes_large(tmp_path):
    decision = router(tmp_path / "missing.joblib").decide("auto", MSGS)
    assert decision.tier == "large"
    assert decision.reason == "no_model"


def test_threshold_zero_routes_small(tmp_path):
    decision = router(artifact(tmp_path, 0.0)).decide("auto", MSGS)
    assert decision.tier == "small"
    assert decision.reason == "model"
    assert 0 <= decision.score <= 1


def test_threshold_one_routes_large(tmp_path):
    decision = router(artifact(tmp_path, 1.01)).decide("auto", MSGS)
    assert decision.tier == "large"

"""Train the router (spec 5.2): uv run python -m scripts.train_router

Compares two Scikit-learn classifiers on the validation split, calibrates the winner, picks the
lowest threshold that keeps routed quality within 95% of always-large, and writes
models/router-v1.joblib, its model card models/router-v1.json, and results/splits.json.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.features import Embedder, build_features
from app.router import FEATURE_VERSION

SEED = 42
LABELS = Path("data/labels.parquet")
ARTIFACT = Path("models/router-v1.joblib")
CARD = Path("models/router-v1.json")
SPLITS = Path("results/splits.json")


def split(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15, stratified on source and label."""
    strata = df["source"] + "_" + df["small_ok"].astype(str)
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=seed)
    rest_strata = rest["source"] + "_" + rest["small_ok"].astype(str)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest_strata, random_state=seed)
    return train, val, test


def routed_quality(p: np.ndarray, small_ok: np.ndarray, t: float) -> float:
    """Quality relative to always-large: a large answer scores 1, a small one scores small_ok."""
    return float(np.mean(np.where(p < t, 1, small_ok)))


def choose_threshold(p: np.ndarray, small_ok: np.ndarray, target: float = 0.95) -> float:
    for t in np.round(np.arange(0, 1.01, 0.01), 2):
        if routed_quality(p, small_ok, t) >= target:
            return float(t)
    return 1.01


def candidates() -> dict:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED
        ),
    }


def features(df: pd.DataFrame, embedder: Embedder) -> np.ndarray:
    return build_features([[{"role": "user", "content": p}] for p in df["prompt"]], embedder)


def main() -> None:
    df = pd.read_parquet(LABELS)
    train, val, test = split(df)
    embedder = Embedder()
    x_train, x_val, x_test = (features(d, embedder) for d in (train, val, test))
    y_train, y_val, y_test = (d["small_ok"].to_numpy() for d in (train, val, test))

    scores = {}
    for name, model in candidates().items():
        model.fit(x_train, y_train)
        scores[name] = float(roc_auc_score(y_val, model.predict_proba(x_val)[:, 1]))
        print(f"{name}: validation ROC-AUC {scores[name]:.3f}")
    chosen = max(scores, key=scores.get)

    calibrated = CalibratedClassifierCV(clone(candidates()[chosen]), method="sigmoid", cv=5)
    calibrated.fit(x_train, y_train)
    p_val = calibrated.predict_proba(x_val)[:, 1]
    threshold = choose_threshold(p_val, y_val)

    # The test split is touched once, here.
    p_test = calibrated.predict_proba(x_test)[:, 1]
    pred_small = (p_test >= threshold).astype(int)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": calibrated, "threshold": threshold, "feature_version": FEATURE_VERSION},
        ARTIFACT,
    )
    card = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "label_balance": float(df["small_ok"].mean()),
        "candidates": {name: {"val_roc_auc": s} for name, s in scores.items()},
        "chosen": chosen,
        "threshold": threshold,
        "val_routed_quality": routed_quality(p_val, y_val, threshold),
        "val_share_small": float(np.mean(p_val >= threshold)),
        "test_roc_auc": float(roc_auc_score(y_test, p_test)),
        "test_precision": float(precision_score(y_test, pred_small, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred_small, zero_division=0)),
        "feature_version": FEATURE_VERSION,
    }
    CARD.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    SPLITS.write_text(
        json.dumps(
            {
                k: d["id"].tolist()
                for k, d in zip(("train", "val", "test"), (train, val, test), strict=True)
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()

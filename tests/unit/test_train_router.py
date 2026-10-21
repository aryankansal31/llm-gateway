import numpy as np
import pandas as pd
import pytest

from scripts.train_router import choose_threshold, routed_quality, split


def synthetic(n: int = 600) -> pd.DataFrame:
    sources = ["dolly", "gsm8k", "mmlu", "humaneval", "mtbench"]
    return pd.DataFrame(
        {
            "id": [f"row-{i}" for i in range(n)],
            "source": [sources[i % 5] for i in range(n)],
            "small_ok": [(i // 5) % 2 for i in range(n)],
        }
    )


def test_split_sizes_and_disjoint():
    train, val, test = split(synthetic())
    assert (len(train), len(val), len(test)) == (420, 90, 90)
    ids = [set(train.id), set(val.id), set(test.id)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_routed_quality_all_large():
    p, ok = np.array([0.9, 0.1]), np.array([0, 1])
    assert routed_quality(p, ok, t=1.01) == 1.0


def test_routed_quality_all_small():
    p, ok = np.array([0.9, 0.1, 0.5]), np.array([0, 1, 1])
    assert routed_quality(p, ok, t=0.0) == pytest.approx(ok.mean())


def test_choose_threshold_lowest_meeting_target():
    p, ok = np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])
    assert choose_threshold(p, ok) == pytest.approx(0.21)

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark import quality_table, summarize


def test_summarize_percentiles():
    assert summarize([100] * 19 + [1000])["p95"] == pytest.approx(145.0)


def test_quality_table_keys():
    test_df = pd.DataFrame({"small_ok": [1, 0, 1, 1]})
    table = quality_table(test_df, np.array([0.9, 0.2, 0.8, 0.1]), threshold=0.5)
    assert set(table) == {"always_small", "always_large", "routed"}
    assert table["always_large"] == 1.0
    assert table["always_small"] == pytest.approx(0.75)
    assert table["routed"] == pytest.approx(1.0)

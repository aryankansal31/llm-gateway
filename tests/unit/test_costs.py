from decimal import Decimal
from pathlib import Path

import pytest

from app.config import load_prices
from app.costs import notional_cost

PRICES = load_prices(Path("config/prices.yaml"))


def test_large_cost():
    assert notional_cost("large", 1_000_000, 1_000_000, PRICES) == Decimal("1.38000000")


def test_small_cost_small_numbers():
    assert notional_cost("small", 500, 200, PRICES) == Decimal("0.00004200")


def test_cache_is_free():
    assert notional_cost("cache", 10_000, 10_000, PRICES) == 0


def test_unknown_tier_raises():
    with pytest.raises(KeyError):
        notional_cost("gpt-9", 1, 1, PRICES)

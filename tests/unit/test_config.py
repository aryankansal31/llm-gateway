from decimal import Decimal
from pathlib import Path

from app.config import Settings, load_prices, load_tiers


def test_load_tiers_reads_all_four_tiers():
    tiers = load_tiers(Path("config/models.yaml"))
    assert set(tiers) == {"small", "large", "large_offline", "judge"}
    assert tiers["small"].model == "llama3.2:3b"


def test_judge_is_different_model_than_large():
    tiers = load_tiers(Path("config/models.yaml"))
    assert tiers["judge"].model != tiers["large"].model


def test_load_prices_returns_decimals():
    prices = load_prices(Path("config/prices.yaml"))
    assert prices["large"].usd_per_1m_output == Decimal("0.79")


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("REDIS_URL", "redis://x")
    settings = Settings(_env_file=None)
    assert settings.rate_limit_per_minute == 60
    assert settings.cache_similarity_threshold == 0.95

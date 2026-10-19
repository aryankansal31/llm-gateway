"""Runtime settings (from the environment) and the YAML tier and price tables."""

from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class TierConfig(BaseModel):
    provider: Literal["ollama", "groq"]
    model: str


class TierPrice(BaseModel):
    usd_per_1m_input: Decimal
    usd_per_1m_output: Decimal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    redis_url: str
    groq_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    models_config_path: Path = Path("config/models.yaml")
    prices_config_path: Path = Path("config/prices.yaml")
    router_model_path: Path = Path("models/router-v1.joblib")
    rate_limit_per_minute: int = 60
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400
    small_max_prompt_tokens: int = 3000
    phoenix_endpoint: str | None = None


def load_tiers(path: Path) -> dict[str, TierConfig]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: TierConfig(**cfg) for name, cfg in data["tiers"].items()}


def load_prices(path: Path) -> dict[str, TierPrice]:
    # Prices are parsed from strings so 0.79 stays exactly Decimal("0.79").
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        name: TierPrice(**{k: Decimal(str(v)) for k, v in cfg.items()})
        for name, cfg in data["tiers"].items()
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Notional cost: what a request would cost at the reference prices in config/prices.yaml."""

from decimal import Decimal

from app.config import TierPrice

_PER_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.00000001")


def notional_cost(
    tier: str, prompt_tokens: int, completion_tokens: int, prices: dict[str, TierPrice]
) -> Decimal:
    if tier == "cache":
        return Decimal(0).quantize(_QUANTUM)
    price = prices[tier]
    cost = (
        prompt_tokens * price.usd_per_1m_input + completion_tokens * price.usd_per_1m_output
    ) / _PER_MILLION
    return cost.quantize(_QUANTUM)

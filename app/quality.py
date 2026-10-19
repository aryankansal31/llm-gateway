"""Cheap, deterministic checks on the small model's answer (spec 5.5)."""

import json
import re

_REFUSAL = re.compile(
    r"^(i can['’]?t|i cannot|i['’]m unable|i am unable|sorry, but i)|as an ai", re.I
)
_REFUSAL_WINDOW = 120


def check_answer(text: str, finish_reason: str | None, wants_json: bool) -> str | None:
    """Return None when the answer is usable, otherwise the reason it is not."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if finish_reason == "length":
        return "truncated"
    if _REFUSAL.search(stripped[:_REFUSAL_WINDOW]):
        return "refusal"
    if wants_json:
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return "invalid_json"
    return None

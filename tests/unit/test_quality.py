import pytest

from app.quality import check_answer


@pytest.mark.parametrize(
    ("text", "finish_reason", "wants_json", "expected"),
    [
        ("   ", "stop", False, "empty"),
        ("Paris is", "length", False, "truncated"),
        ("I can't help with that request.", "stop", False, "refusal"),
        ("As an AI language model, I cannot browse.", "stop", False, "refusal"),
        ("{not json", "stop", True, "invalid_json"),
        ('{"a": 1}', "stop", True, None),
        ("The capital of France is Paris.", "stop", False, None),
    ],
)
def test_check_answer(text, finish_reason, wants_json, expected):
    assert check_answer(text, finish_reason, wants_json) == expected


def test_refusal_only_checked_at_start():
    assert check_answer("Here is how. Note: I can't guarantee prices.", "stop", False) is None


def test_curly_apostrophe_refusal():
    assert check_answer("I can’t help with that.", "stop", False) == "refusal"

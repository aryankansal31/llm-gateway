from types import SimpleNamespace

from scripts.llm_cache import CachedLLM

MSGS = [{"role": "user", "content": "hi"}]


class FakeOpenAI:
    """Counts calls to chat.completions.create and returns a fixed answer."""

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        )


def test_cache_hit_avoids_second_call(tmp_path):
    fake = FakeOpenAI()
    llm = CachedLLM(tmp_path / "c.sqlite", {"small": (fake, "m-small")})
    first = llm.complete("small", MSGS)
    second = llm.complete("small", MSGS)
    assert fake.calls == 1
    assert first["text"] == second["text"] == "hello"


def test_cache_key_includes_model(tmp_path):
    fake = FakeOpenAI()
    llm = CachedLLM(tmp_path / "c.sqlite", {"a": (fake, "model-a"), "b": (fake, "model-b")})
    llm.complete("a", MSGS)
    llm.complete("b", MSGS)
    assert fake.calls == 2

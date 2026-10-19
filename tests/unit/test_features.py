import math

import numpy as np
import pytest

from app.features import HANDCRAFTED_NAMES, Embedder, build_features, handcrafted, last_user_text


class FakeEmbedder(Embedder):
    def __init__(self):
        pass

    def embed(self, texts):
        return np.ones((len(texts), 384), dtype=np.float32) / math.sqrt(384)


def feature(values, name):
    return values[HANDCRAFTED_NAMES.index(name)]


def test_handcrafted_shape_and_order():
    values = handcrafted([{"role": "user", "content": "What is 12 x 7? Explain why."}])
    assert values.shape == (7,)
    assert feature(values, "n_questions") == 1
    assert feature(values, "has_math_symbols") == 1
    assert feature(values, "has_reasoning_words") == 1
    assert feature(values, "n_turns") == 1


def test_code_detection():
    values = handcrafted([{"role": "user", "content": "Fix this:\n```\nprint(1\n```"}])
    assert feature(values, "has_code") == 1


def test_last_user_text_skips_assistant():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "another reply"},
    ]
    assert last_user_text(msgs) == "second"


def test_build_features_shape():
    batch = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "prove it"}]]
    assert build_features(batch, FakeEmbedder()).shape == (2, 391)


@pytest.mark.slow
def test_real_embedder_normalised():
    norms = np.linalg.norm(Embedder().embed(["hi"]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)

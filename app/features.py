"""Router features, shared by training (scripts/train_router.py) and serving (app/router.py).

Messages are OpenAI-style dicts whose `content` is already plain text.
"""

import math
import re

import numpy as np

EMBED_DIM = 384
HANDCRAFTED_NAMES = [
    "n_tokens_est",
    "n_questions",
    "has_code",
    "digit_ratio",
    "has_math_symbols",
    "has_reasoning_words",
    "n_turns",
]

_CODE = re.compile(r"```|\bdef |\bclass |\bfunction |;\n")
_MATH = re.compile(r"[=+*/^√∑∫]|\d+\s*[x×]\s*\d+")
_REASONING = re.compile(r"prove|step by step|explain why|compare|derive|analy[sz]e", re.I)


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def handcrafted(messages: list[dict]) -> np.ndarray:
    text = last_user_text(messages)
    all_text = "\n".join(m["content"] for m in messages)
    digits = sum(c.isdigit() for c in text)
    return np.array(
        [
            estimate_tokens(all_text),
            text.count("?"),
            bool(_CODE.search(text)),
            digits / len(text) if text else 0.0,
            bool(_MATH.search(text)),
            bool(_REASONING.search(text)),
            sum(m["role"] == "user" for m in messages),
        ],
        dtype=np.float32,
    )


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # imported here: loading it is slow

        self.model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_features(messages_batch: list[list[dict]], embedder: Embedder) -> np.ndarray:
    embeddings = embedder.embed([last_user_text(m) for m in messages_batch])
    extra = np.stack([handcrafted(m) for m in messages_batch])
    return np.hstack([embeddings, extra]).astype(np.float32)

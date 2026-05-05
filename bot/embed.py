"""Thin wrapper around OpenAI's embeddings API.

We use text-embedding-3-small (1536 dim) — cheap, multilingual, more than
strong enough for matching short Russian/English queries to grant descriptions.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from openai import OpenAI


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Embeddings require an OpenAI key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Return an L2-normalized (N, EMBEDDING_DIM) matrix.

    Normalization up front means cosine similarity is just a dot product
    later, which is the hot path on every user query.
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    client = _get_client()
    # The API accepts a list of strings; one round trip for all chunks at
    # build time is fine for our scale.
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    vectors = np.array(
        [item.embedding for item in resp.data], dtype=np.float32
    )
    # L2 normalize each row.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_query(text: str) -> np.ndarray:
    """Return a single L2-normalized (EMBEDDING_DIM,) vector."""
    return embed_texts([text])[0]

"""Retrieval over the prebuilt grant embeddings.

The index is loaded once per cold start and held in module globals. Lookup
is a single dot-product against an L2-normalized matrix — for our scale
(tens to low thousands of grants) this is faster than spinning up FAISS,
and ships in zero extra dependencies.

If the corpus ever grows past ~50k rows, drop in faiss-cpu by replacing
`_topk_cosine` with an `IndexFlatIP` lookup; the rest of the module stays
the same.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bot.embed import embed_query


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
INDEX_PATH = ROOT / "data" / "index.npz"


@dataclass
class Retrieved:
    row_id: int
    text: str
    grant: dict[str, Any]
    score: float


_lock = threading.Lock()
_chunks: list[dict[str, Any]] | None = None
_matrix: np.ndarray | None = None


def _ensure_loaded() -> tuple[list[dict[str, Any]], np.ndarray]:
    global _chunks, _matrix
    with _lock:
        if _chunks is None or _matrix is None:
            if not CHUNKS_PATH.exists() or not INDEX_PATH.exists():
                raise FileNotFoundError(
                    "Index not built. Run `python scripts/build_index.py` first."
                )
            _chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            with np.load(INDEX_PATH) as data:
                _matrix = data["embeddings"].astype(np.float32)
            if _matrix.shape[0] != len(_chunks):
                raise RuntimeError(
                    f"Index/chunks size mismatch: {_matrix.shape[0]} vs {len(_chunks)}"
                )
    return _chunks, _matrix


def reload_index() -> None:
    """Force the next retrieve() call to reread chunks.json and index.npz.

    Used by the /reload admin command so we can swap data without redeploying.
    """
    global _chunks, _matrix
    with _lock:
        _chunks = None
        _matrix = None


def retrieve(query: str, top_k: int = 5) -> list[Retrieved]:
    """Return the top_k grants most relevant to `query`, highest score first."""
    chunks, matrix = _ensure_loaded()
    if not query.strip() or matrix.shape[0] == 0:
        return []

    q_vec = embed_query(query)  # already L2-normalized
    # Both sides normalized -> dot product == cosine similarity
    scores = matrix @ q_vec  # shape (N,)
    k = min(top_k, scores.shape[0])
    # argpartition for top-k, then sort just those k
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    return [
        Retrieved(
            row_id=chunks[i]["row_id"],
            text=chunks[i]["text"],
            grant=chunks[i]["grant"],
            score=float(scores[i]),
        )
        for i in top_idx
    ]


def grant_count() -> int:
    chunks, _ = _ensure_loaded()
    return len(chunks)

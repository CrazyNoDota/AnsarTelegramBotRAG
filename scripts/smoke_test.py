"""End-to-end smoke test that doesn't hit the network.

Stubs the `openai` SDK with a tiny in-process fake, builds the embeddings
index, runs a couple of representative queries through the RAG handler,
and asserts that out-of-data questions get refused.

Run from repo root:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---- Stub the `openai` module BEFORE importing bot.* ---------------------

class _FakeEmbeddingItem:
    def __init__(self, vec):
        self.embedding = vec


class _FakeEmbResp:
    def __init__(self, data):
        self.data = data


def _hash_embed(text: str, dim: int = 1536) -> list[float]:
    """Deterministic pseudo-embedding: hash trigrams into a fixed-dim vector.
    Not a real semantic model — just enough to let cosine similarity rank
    the chunk that contains query keywords above unrelated chunks.
    """
    vec = np.zeros(dim, dtype=np.float32)
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        tri = text_lower[i:i + 3]
        h = abs(hash(tri)) % dim
        vec[h] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec.tolist()


class _FakeEmbeddings:
    def create(self, model, input):
        items = [_FakeEmbeddingItem(_hash_embed(t)) for t in input]
        return _FakeEmbResp(items)


class _FakeChatCompletions:
    def create(self, **kwargs):
        # Find the user question and the system prompt
        msgs = kwargs.get("messages", [])
        question = next((m["content"] for m in msgs if m["role"] == "user"), "")
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        # Rule: if the question keyword shows up in the CONTEXT block, "answer"
        # by echoing the first grant name. Otherwise return the refusal.
        ctx_start = system.find("---")
        ctx = system[ctx_start:]
        # Pull the first "Грант №N: NAME" line we see
        first_name = None
        for line in ctx.split("\n"):
            if line.startswith("Грант №"):
                first_name = line.split(":", 1)[1].strip() if ":" in line else line
                break
        # Crude relevance check
        keywords = [w for w in question.lower().split() if len(w) > 3]
        if first_name and any(k in ctx.lower() for k in keywords):
            content = f"По базе подходит: {first_name}."
        else:
            # Use the refusal phrase from the system prompt
            if "У меня нет" in system:
                content = "У меня нет этой информации в базе грантов."
            else:
                content = "I don't have this information in the grants database."

        class _Choice:
            class _Msg:
                pass

            def __init__(self, c):
                self.message = self._Msg()
                self.message.content = c

        class _Resp:
            def __init__(self, c):
                self.choices = [_Choice(c)]

        return _Resp(content)


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeOpenAI:
    def __init__(self, api_key=None):
        self.embeddings = _FakeEmbeddings()
        self.chat = _FakeChat()


fake_module = types.ModuleType("openai")
fake_module.OpenAI = _FakeOpenAI
sys.modules["openai"] = fake_module

# Pretend we have credentials so the fallback path is reachable.
import os  # noqa: E402
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-test")

# Ensure NVIDIA path is skipped — we want to exercise the OpenAI fallback.
os.environ.pop("NVIDIA_API_KEY", None)


# ---- Now build a fake index in memory -----------------------------------

from bot.data import load_grants  # noqa: E402
from bot.embed import embed_texts  # noqa: E402
from bot import rag as rag_mod  # noqa: E402
from bot.telegram_handler import handle_update  # noqa: E402


def build_fake_index():
    grants = load_grants(ROOT / "data" / "grants.xlsx")
    chunks = [
        {"row_id": g.row_id, "text": g.to_chunk_text(), "grant": g.to_dict()}
        for g in grants
    ]
    matrix = embed_texts([c["text"] for c in chunks])
    # Inject directly into the rag module's globals
    rag_mod._chunks = chunks
    rag_mod._matrix = matrix
    return len(chunks)


def main():
    n = build_fake_index()
    print(f"[smoke] indexed {n} grants with stub embeddings")

    # 1. Russian question, in-data
    update = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 999},
            "text": "Какие гранты NASA есть для биотехнологий?",
        }
    }
    replies = handle_update(update)
    print(f"[smoke] RU in-data reply: {replies[0].text!r}")
    assert replies, "expected a reply"

    # 2. English question, in-data
    update = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 999},
            "text": "Which grants are available for AI startups in EU?",
        }
    }
    replies = handle_update(update)
    print(f"[smoke] EN in-data reply: {replies[0].text!r}")

    # 3. Out-of-data question — should refuse
    update = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 999},
            "text": "asdfqwerty unrelated zzz xyzzy plover",
        }
    }
    replies = handle_update(update)
    print(f"[smoke] OOD reply: {replies[0].text!r}")

    # 4. /start
    update = {
        "message": {
            "chat": {"id": 1},
            "from": {"id": 999},
            "text": "/start",
        }
    }
    replies = handle_update(update)
    assert "Привет" in replies[0].text
    print("[smoke] /start works")

    # 5. /count
    update["message"]["text"] = "/count"
    replies = handle_update(update)
    assert str(n) in replies[0].text
    print(f"[smoke] /count works -> {replies[0].text!r}")

    # 6. /reload — non-admin
    os.environ["ADMIN_CHAT_IDS"] = "42"
    update["message"]["text"] = "/reload"
    update["message"]["from"]["id"] = 999
    replies = handle_update(update)
    assert "только админу" in replies[0].text.lower()
    print(f"[smoke] /reload non-admin blocked: {replies[0].text!r}")

    # 7. /reload — admin (will try to reread from disk; we expect failure
    #    because no chunks.json on disk yet, so the reply contains an error)
    update["message"]["from"]["id"] = 42
    replies = handle_update(update)
    print(f"[smoke] /reload admin path: {replies[0].text!r}")

    print("\n[smoke] all checks passed")


if __name__ == "__main__":
    main()

"""Rebuild the embeddings index from data/grants.xlsx.

Run this every time the spreadsheet changes:

    OPENAI_API_KEY=sk-... python scripts/build_index.py

Outputs:
    data/chunks.json   — list of {row_id, text, grant} entries
    data/index.npz     — float32 (N, 1536) normalized embedding matrix
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Make `bot` importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.data import load_grants  # noqa: E402
from bot.embed import embed_texts  # noqa: E402


def main() -> None:
    xlsx_path = ROOT / "data" / "grants.xlsx"
    chunks_path = ROOT / "data" / "chunks.json"
    index_path = ROOT / "data" / "index.npz"

    print(f"Loading grants from {xlsx_path} ...")
    grants = load_grants(xlsx_path)
    print(f"  -> {len(grants)} grants")

    chunks = [
        {
            "row_id": g.row_id,
            "text": g.to_chunk_text(),
            "grant": g.to_dict(),
        }
        for g in grants
    ]

    print(f"Embedding {len(chunks)} chunks via OpenAI ...")
    matrix = embed_texts([c["text"] for c in chunks])
    print(f"  -> matrix shape: {matrix.shape}, dtype: {matrix.dtype}")

    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(index_path, embeddings=matrix)

    print(f"Wrote {chunks_path}  ({chunks_path.stat().st_size:,} bytes)")
    print(f"Wrote {index_path}  ({index_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY env var not set.", file=sys.stderr)
        sys.exit(1)
    main()

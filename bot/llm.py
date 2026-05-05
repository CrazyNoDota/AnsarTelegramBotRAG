"""LLM client with Qwen-on-NVIDIA primary and OpenAI fallback.

The system prompt is the main hallucination control: it explicitly tells
the model that the only allowed source of facts is the CONTEXT block, and
that anything not present must be answered with a fixed refusal phrase.

Language handling: we don't ask the model to translate or guess — we
detect Cyrillic in the user's question and tell the model which language
to reply in. This keeps the system simple and predictable.
"""
from __future__ import annotations

import os
from typing import Iterable

import requests
from openai import OpenAI

from bot.rag import Retrieved


# NVIDIA NIM hosts a number of OpenAI-compatible models. The exact model id
# can change — keep this configurable via env so we can swap without a code
# push if NVIDIA renames or retires it.
NVIDIA_BASE_URL = os.environ.get(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
)
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "qwen/qwen3-235b-a22b")
OPENAI_FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

REFUSAL_RU = "У меня нет этой информации в базе грантов."
REFUSAL_EN = "I don't have this information in the grants database."


SYSTEM_PROMPT_TEMPLATE = """Ты — ассистент по базе грантовых программ.

ЖЁСТКИЕ ПРАВИЛА (нарушать запрещено):
1. Отвечай ТОЛЬКО на основе блока CONTEXT ниже. Не используй внешние знания.
2. Если в CONTEXT нет ответа — ответь точно фразой:
   "{refusal}"
   Не выдумывай, не догадывайся, не добавляй "вероятно".
3. Цитируй конкретные гранты по их названию (и номеру строки) когда отвечаешь.
4. Если пользователь спросил по-русски — отвечай по-русски. Если по-английски — отвечай по-английски.
5. Будь краток. Если фактов мало — короткий ответ. Не пересказывай весь CONTEXT.

CONTEXT (это вся доступная тебе информация о грантах):
---
{context}
---
"""


def _is_cyrillic(text: str) -> bool:
    """Cheap language detector — 'is there Cyrillic?'.

    Good enough for ru/en routing. Replace with langdetect if more languages
    show up later.
    """
    return any("Ѐ" <= ch <= "ӿ" for ch in text)


def _build_context_block(hits: Iterable[Retrieved]) -> str:
    parts: list[str] = []
    for h in hits:
        parts.append(h.text)
        parts.append("---")
    return "\n".join(parts).rstrip("-\n")


def _call_nvidia(messages: list[dict], api_key: str) -> str:
    """Call NVIDIA's OpenAI-compatible endpoint. Raises on non-2xx."""
    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "model": NVIDIA_MODEL,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.2,  # low — we want grounded, not creative
            "top_p": 0.9,
            "stream": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    return body["choices"][0]["message"]["content"].strip()


def _call_openai(messages: list[dict], api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_FALLBACK_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


def answer(question: str, hits: list[Retrieved]) -> str:
    """Generate a grounded answer to `question` using the retrieved hits.

    Tries NVIDIA Qwen first (free tier); on any error falls back to OpenAI.
    Both keys can be present — the fallback is silent. If neither is set,
    raises so the bot can surface a configuration problem.
    """
    refusal = REFUSAL_RU if _is_cyrillic(question) else REFUSAL_EN
    if not hits:
        return refusal

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        refusal=refusal,
        context=_build_context_block(hits),
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if nvidia_key:
        try:
            return _call_nvidia(messages, nvidia_key)
        except Exception as exc:  # noqa: BLE001 — we want to fall back on anything
            print(f"[llm] NVIDIA call failed, falling back to OpenAI: {exc}")

    if openai_key:
        return _call_openai(messages, openai_key)

    raise RuntimeError(
        "No LLM credentials available. Set NVIDIA_API_KEY and/or OPENAI_API_KEY."
    )

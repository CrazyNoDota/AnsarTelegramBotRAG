"""Telegram update handler — pure logic, no transport.

Takes a parsed Telegram update dict and returns a list of messages to send.
This split keeps the Vercel serverless entrypoint tiny and the logic
unit-testable without spinning up a real bot.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from bot.rag import retrieve, reload_index, grant_count
from bot.llm import answer


WELCOME = (
    "Welcome! I'm your personal university admissions advisor.\n\n"
    "Our company helps students find scholarships and grants, prepare documents, "
    "and successfully apply to their dream university. We have {n} programs in our database!\n\n"
    "Where do you want to start? Tell me:\n"
    "• Which country are you looking to study in?\n"
    "• What field or subject interests you?\n"
    "• Or just ask — for example:\n"
    "  \"What scholarships are available for IT in the US?\"\n"
    "  \"Are there grants for medical students in Europe?\"\n\n"
    "I'm here to help you reach your dream university! Feel free to write in any language."
)

HELP = (
    "I'm a university admissions and scholarship advisor.\n\n"
    "Just tell me about your goals — country, field of study, degree level — "
    "and I'll find the best matching programs from our database.\n\n"
    "You can write in any language and I'll reply in kind.\n\n"
    "Commands:\n"
    "/start — welcome message\n"
    "/help — this help\n"
    "/count — number of programs in the database\n"
    "/reload — refresh the database index (admin only)"
)


@dataclass
class Reply:
    chat_id: int
    text: str


def _parse_admin_ids() -> set[int]:
    raw = os.environ.get("ADMIN_CHAT_IDS", "")
    out: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def handle_update(update: dict) -> list[Reply]:
    """Top-level dispatcher.

    Telegram sends many update types (messages, edits, callback queries...).
    We only care about plain text messages here; everything else is silently
    ignored so the bot doesn't crash on stickers, photos, or button presses.
    """
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return []

    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return []

    text = (msg.get("text") or "").strip()
    if not text:
        return []

    user_id = (msg.get("from") or {}).get("id")

    # ---- commands ----
    if text.startswith("/"):
        cmd = text.split()[0].split("@")[0].lower()  # strip /cmd@botname

        if cmd == "/start":
            return [Reply(chat_id, WELCOME.format(n=grant_count()))]

        if cmd == "/help":
            return [Reply(chat_id, HELP)]

        if cmd == "/count":
            return [Reply(chat_id, f"We currently have {grant_count()} scholarship programs in our database.")]

        if cmd == "/reload":
            admins = _parse_admin_ids()
            if not admins:
                return [Reply(chat_id, "ADMIN_CHAT_IDS is not configured in the environment.")]
            if user_id not in admins:
                return [Reply(chat_id, "This command is available to admins only.")]
            try:
                reload_index()
                return [Reply(chat_id, f"Database refreshed. Programs: {grant_count()}.")]
            except Exception as exc:  # noqa: BLE001
                return [Reply(chat_id, f"Failed to reload: {exc}")]

        return [Reply(chat_id, "Unknown command. Just tell me about your goals and I'll find the right program! /help for the command list.")]

    # ---- regular question ----
    try:
        hits = retrieve(text, top_k=5)
        reply_text = answer(text, hits)
    except Exception as exc:  # noqa: BLE001
        reply_text = f"Something went wrong ({type(exc).__name__}). Please try again in a moment."
        print(f"[handler] error processing message: {exc!r}")

    return [Reply(chat_id, reply_text)]

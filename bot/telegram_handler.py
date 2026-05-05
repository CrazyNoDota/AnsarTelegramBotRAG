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


WELCOME_RU = (
    "Привет! 🎓 Я твой личный консультант по поступлению в университет.\n\n"
    "Наша компания помогает студентам найти стипендии и гранты, оформить документы "
    "и успешно подать заявку в университет мечты. В нашей базе уже {n} программ!\n\n"
    "С чего начнём? Расскажи:\n"
    "• В какую страну хочешь поступить?\n"
    "• Какую специальность рассматриваешь?\n"
    "• Или просто спроси — например:\n"
    "  «Какие стипендии есть для IT в США?»\n"
    "  «What scholarships are available for engineering?»\n\n"
    "Я здесь, чтобы помочь тебе добраться до университета мечты! 🚀"
)

HELP_RU = (
    "Я консультант по поступлению в университет и поиску стипендий.\n\n"
    "Просто напиши мне о своих целях — страна, специальность, уровень образования — "
    "и я найду подходящие программы из нашей базы.\n\n"
    "Команды:\n"
    "/start — приветствие\n"
    "/help — эта справка\n"
    "/count — сколько программ в базе\n"
    "/reload — обновить базу (только для админа)"
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
            return [Reply(chat_id, WELCOME_RU.format(n=grant_count()))]

        if cmd == "/help":
            return [Reply(chat_id, HELP_RU)]

        if cmd == "/count":
            return [Reply(chat_id, f"В нашей базе сейчас {grant_count()} стипендиальных программ.")]

        if cmd == "/reload":
            admins = _parse_admin_ids()
            if not admins:
                return [Reply(chat_id, "ADMIN_CHAT_IDS не настроены в окружении.")]
            if user_id not in admins:
                return [Reply(chat_id, "Команда доступна только админу.")]
            try:
                reload_index()
                return [Reply(chat_id, f"База обновлена. Программ: {grant_count()}.")]
            except Exception as exc:  # noqa: BLE001
                return [Reply(chat_id, f"Не удалось перезагрузить: {exc}")]

        return [Reply(chat_id, "Не знаю такой команды. Просто напиши мне о своих целях — и я помогу найти стипендию! /help — список команд.")]

    # ---- regular question ----
    try:
        hits = retrieve(text, top_k=5)
        reply_text = answer(text, hits)
    except Exception as exc:  # noqa: BLE001
        # Surface enough to debug from the user's chat without leaking secrets.
        reply_text = f"Внутренняя ошибка: {type(exc).__name__}. Попробуй ещё раз позже."
        print(f"[handler] error processing message: {exc!r}")

    return [Reply(chat_id, reply_text)]

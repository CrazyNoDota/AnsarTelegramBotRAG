"""Telegram update handler — pure logic, no transport.

Accepts a `language` parameter ("en" or "ru") that controls both the static
message language and the LLM reply language, letting one codebase power two
separate Telegram bots.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from bot.rag import retrieve, reload_index, grant_count
from bot.llm import answer


_STATIC: dict[str, dict[str, str]] = {
    "en": {
        "welcome": (
            "Welcome! I'm your Pakistan funding advisor.\n\n"
            "I help organizations, startups, universities, and students in Pakistan "
            "find grants and funding opportunities. We have {n} programs in our database!\n\n"
            "Tell me about yourself:\n"
            "• Are you an individual (student / researcher) or an institution?\n"
            "• What sector or field are you in?\n"
            "• Or just ask — for example:\n"
            '  "What grants are available for tech startups in Pakistan?"\n'
            '  "Are there research grants for engineering students?"\n'
            '  "Which programs have deadlines coming up soon?"\n\n'
            "Write in any language — I'll always reply in English."
        ),
        "help": (
            "I'm your Pakistan funding advisor.\n\n"
            "Tell me about your goals — individual or institution, sector, "
            "funding amount — and I'll match you with the best programs from our database.\n\n"
            "I can help you:\n"
            "• Find relevant grants and funding programs\n"
            "• Check your eligibility for specific opportunities\n"
            "• Understand deadlines and requirements\n"
            "• Get guidance on how to apply\n\n"
            "I always reply in English.\n\n"
            "Commands:\n"
            "/start — welcome message\n"
            "/help — this help\n"
            "/count — number of programs in the database\n"
            "/reload — refresh the database index (admin only)"
        ),
        "count": "We currently have {n} funding programs in our database.",
        "reload_ok": "Database refreshed. Programs: {n}.",
        "reload_fail": "Failed to reload: {exc}",
        "no_admin_env": "ADMIN_CHAT_IDS is not configured in the environment.",
        "not_admin": "This command is available to admins only.",
        "unknown_cmd": (
            "Unknown command. Tell me about your funding goals and I'll find the right program! "
            "/help for the command list."
        ),
        "error": "Something went wrong ({exc_type}). Please try again in a moment.",
    },
    "ru": {
        "welcome": (
            "Добро пожаловать! Я твой консультант по финансированию в Пакистане.\n\n"
            "Я помогаю организациям, стартапам, университетам и студентам в Пакистане "
            "находить гранты и возможности финансирования. В нашей базе {n} программ!\n\n"
            "Расскажи о себе:\n"
            "• Ты физическое лицо (студент / исследователь) или организация?\n"
            "• В какой сфере или области работаешь?\n"
            "• Или просто задай вопрос — например:\n"
            '  «Какие гранты есть для технологических стартапов в Пакистане?»\n'
            '  «Есть ли исследовательские гранты для студентов-инженеров?»\n'
            '  «Какие программы заканчиваются в ближайшее время?»\n\n'
            "Пиши на любом языке — я всегда отвечу на русском."
        ),
        "help": (
            "Я твой консультант по финансированию в Пакистане.\n\n"
            "Расскажи о своих целях — физическое лицо или организация, сфера, "
            "объём финансирования — и я подберу лучшие программы из нашей базы данных.\n\n"
            "Я помогу:\n"
            "• Найти актуальные гранты и программы финансирования\n"
            "• Проверить твою eligibility для конкретных возможностей\n"
            "• Разобраться в сроках и требованиях\n"
            "• Получить руководство по подаче заявки\n\n"
            "Я всегда отвечаю на русском языке.\n\n"
            "Команды:\n"
            "/start — приветственное сообщение\n"
            "/help — эта справка\n"
            "/count — количество программ в базе данных\n"
            "/reload — обновить индекс базы данных (только для администраторов)"
        ),
        "count": "В нашей базе данных сейчас {n} программ финансирования.",
        "reload_ok": "База данных обновлена. Программ: {n}.",
        "reload_fail": "Ошибка обновления: {exc}",
        "no_admin_env": "ADMIN_CHAT_IDS не настроен в переменных окружения.",
        "not_admin": "Эта команда доступна только администраторам.",
        "unknown_cmd": (
            "Неизвестная команда. Расскажи о своих целях — я найду подходящую программу! "
            "/help для списка команд."
        ),
        "error": "Что-то пошло не так ({exc_type}). Попробуй ещё раз через мгновение.",
    },
}


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


def handle_update(update: dict, language: str = "en") -> list[Reply]:
    """Top-level dispatcher.

    language: "en" or "ru" — controls static messages and LLM reply language.
    Telegram sends many update types; we only handle plain text messages.
    """
    lang = language if language in _STATIC else "en"
    s = _STATIC[lang]

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

    if text.startswith("/"):
        cmd = text.split()[0].split("@")[0].lower()

        if cmd == "/start":
            return [Reply(chat_id, s["welcome"].format(n=grant_count()))]

        if cmd == "/help":
            return [Reply(chat_id, s["help"])]

        if cmd == "/count":
            return [Reply(chat_id, s["count"].format(n=grant_count()))]

        if cmd == "/reload":
            admins = _parse_admin_ids()
            if not admins:
                return [Reply(chat_id, s["no_admin_env"])]
            if user_id not in admins:
                return [Reply(chat_id, s["not_admin"])]
            try:
                reload_index()
                return [Reply(chat_id, s["reload_ok"].format(n=grant_count()))]
            except Exception as exc:  # noqa: BLE001
                return [Reply(chat_id, s["reload_fail"].format(exc=exc))]

        return [Reply(chat_id, s["unknown_cmd"])]

    try:
        hits = retrieve(text, top_k=5)
        reply_text = answer(text, hits, language=lang)
    except Exception as exc:  # noqa: BLE001
        reply_text = s["error"].format(exc_type=type(exc).__name__)
        print(f"[handler] error processing message: {exc!r}")

    return [Reply(chat_id, reply_text)]

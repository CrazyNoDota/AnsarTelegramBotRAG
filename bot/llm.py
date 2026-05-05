"""LLM client — NVIDIA NIM primary, OpenAI fallback.

Two language modes:
  "en" — always respond in English; translate any Russian/other context.
  "ru" — always respond in Russian; translate any English/other context.

The system prompt is the main behavior controller: topic restriction,
fact grounding, language lock, and conversation style.
"""
from __future__ import annotations

import os
from typing import Iterable

import requests
from openai import OpenAI

from bot.rag import Retrieved


NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "qwen/qwen3-235b-a22b")
OPENAI_FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

_REFUSALS = {
    "en": (
        "Great question! I help organizations, startups, universities, and students "
        "find grants and funding opportunities. Tell me more — are you an individual or an institution? "
        "What sector or field are you in? Let's find the right funding for you!"
    ),
    "ru": (
        "Отличный вопрос! Я помогаю организациям, стартапам, университетам и студентам "
        "находить гранты и возможности финансирования. Расскажи подробнее — ты физическое лицо "
        "или организация? В какой сфере работаешь? Давай найдём подходящий грант!"
    ),
}

_SYSTEM_EN = """\
You are an enthusiastic and knowledgeable funding advisor helping organizations, startups, \
universities, and students find grants and funding opportunities.

You help users:
• Search and identify relevant grants and funding programs
• Understand which programs they are eligible to apply for
• Track application deadlines and requirements
• Navigate the application process step by step
• Find both institution-level funding (universities, organizations, startups) \
and individual-level funding (students, researchers)

STRICT RULES:
1. TOPIC: You only discuss grants, funding, scholarships, and related application topics \
for grant applicants. For anything off-topic, warmly redirect: \
"I'm your funding advisor! Ask me about grants, funding programs, or how to apply."
2. LANGUAGE: Always respond in English — even if the user writes in another language. \
If retrieved context contains Russian or other languages, translate it to English in your reply.
3. FACTS FROM CONTEXT ONLY: Specific details (amounts, deadlines, eligibility criteria, \
countries) must come ONLY from the CONTEXT block below. Never invent or assume details.
4. WHEN CONTEXT LACKS THE ANSWER: Don't refuse coldly. Ask a clarifying question instead — \
e.g., "Are you an individual or an institution?", "What sector are you in?", \
"What funding amount are you looking for?", "What is your field of study?"
5. STYLE: Be warm, professional, and encouraging. Ask follow-up questions to understand \
the user's profile and goals. Show genuine interest in helping them succeed. \
NEVER start with "Hello", "Hi", "Hey", or any greeting — jump straight into the answer.
6. CONCISE: Pick the most relevant programs from CONTEXT — 3 to 5 with key highlights. \
Don't dump everything.

CONTEXT (funding programs in our database):
---
{context}
---
"""

_SYSTEM_RU = """\
Ты энергичный и опытный консультант по финансированию, который помогает организациям, \
стартапам, университетам и студентам находить гранты и возможности финансирования.

Ты помогаешь пользователям:
• Искать и находить актуальные гранты и программы финансирования
• Понять, на какие программы они могут претендовать
• Отслеживать сроки подачи заявок и требования
• Пройти через процесс подачи заявки шаг за шагом
• Найти финансирование как для организаций (университеты, НКО, стартапы), \
так и для физических лиц (студенты, исследователи)

СТРОГИЕ ПРАВИЛА:
1. ТЕМА: Ты обсуждаешь только гранты, финансирование, стипендии и связанные с этим темы \
для заявителей. Если тема другая — тепло перенаправь: \
«Я твой консультант по финансированию! Спроси меня о грантах или о том, как подать заявку.»
2. ЯЗЫК: Всегда отвечай на русском языке — даже если пользователь пишет на другом языке. \
Если в контексте есть английский или другой язык — переводи на русский в своём ответе.
3. ТОЛЬКО ФАКТЫ ИЗ КОНТЕКСТА: Конкретные детали (суммы, сроки, критерии отбора, страны) \
должны браться ТОЛЬКО из блока КОНТЕКСТ ниже. Никогда не придумывай и не предполагай детали.
4. ЕСЛИ КОНТЕКСТ НЕ СОДЕРЖИТ ОТВЕТА: Не отказывай холодно. Задай уточняющий вопрос — \
например, «Вы физическое лицо или организация?», «В какой сфере работаете?», \
«Какой объём финансирования вам нужен?», «Какая у вас специальность?»
5. СТИЛЬ: Будь тёплым, профессиональным и воодушевляющим. Задавай уточняющие вопросы, \
чтобы понять профиль и цели пользователя. Проявляй искренний интерес в том, чтобы помочь. \
НИКОГДА не начинай с «Привет», «Здравствуйте» или любого другого приветствия — \
сразу переходи к ответу или вопросу.
6. КРАТКО: Выбирай наиболее релевантные программы из КОНТЕКСТА — 3–5 с ключевыми \
особенностями. Не перечисляй всё подряд.

КОНТЕКСТ (программы финансирования в нашей базе данных):
---
{context}
---
"""

_SYSTEM_PROMPTS = {"en": _SYSTEM_EN, "ru": _SYSTEM_RU}


def _build_context_block(hits: Iterable[Retrieved]) -> str:
    parts: list[str] = []
    for h in hits:
        parts.append(h.text)
        parts.append("---")
    return "\n".join(parts).rstrip("-\n")


def _call_nvidia(messages: list[dict], api_key: str) -> str:
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
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=8,
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
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


def answer(question: str, hits: list[Retrieved], language: str = "en") -> str:
    """Generate a grounded answer using retrieved hits.

    language: "en" (English bot) or "ru" (Russian bot).
    Tries NVIDIA first, falls back to OpenAI on any error.
    """
    lang = language if language in _SYSTEM_PROMPTS else "en"
    refusal = _REFUSALS[lang]

    if not hits:
        return refusal

    system_prompt = _SYSTEM_PROMPTS[lang].format(context=_build_context_block(hits))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if nvidia_key:
        try:
            return _call_nvidia(messages, nvidia_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] NVIDIA call failed ({type(exc).__name__}: {exc}), falling back to OpenAI")

    if openai_key:
        try:
            return _call_openai(messages, openai_key)
        except Exception as exc:
            print(f"[llm] OpenAI call failed ({type(exc).__name__}: {exc})")
            raise

    raise RuntimeError("No LLM credentials available. Set NVIDIA_API_KEY and/or OPENAI_API_KEY.")

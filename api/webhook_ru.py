"""Vercel serverless entrypoint — Russian bot.

Receives Telegram webhook POSTs and always replies in Russian.
Reads TELEGRAM_BOT_TOKEN_RU.

Set the webhook after deploy:
    curl -F "url=https://<your-domain>/api/webhook_ru" \
        https://api.telegram.org/bot<RU_TOKEN>/setWebhook
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.telegram_handler import handle_update, Reply  # noqa: E402


def _send(token: str, reply: Reply) -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": reply.chat_id,
                "text": reply.text,
                "disable_web_page_preview": False,
            },
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[webhook_ru] failed to send message: {exc!r}")


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok (ru)")

    def do_POST(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN_RU")
        if not token:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"TELEGRAM_BOT_TOKEN_RU not configured")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            update = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            print(f"[webhook_ru] bad payload: {exc!r}")
            self.send_response(400)
            self.end_headers()
            return

        try:
            replies = handle_update(update, language="ru")
            for r in replies:
                _send(token, r)
        except Exception as exc:  # noqa: BLE001
            print(f"[webhook_ru] handler crashed: {exc!r}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

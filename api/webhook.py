"""Vercel serverless entrypoint — receives Telegram webhook POSTs.

Vercel routes any POST to /api/webhook to this BaseHTTPRequestHandler.
Keep this file thin; all logic lives in `bot.telegram_handler`.

Set the webhook once after deploy:
    curl -F "url=https://<your-vercel-domain>/api/webhook" \
        https://api.telegram.org/bot<TOKEN>/setWebhook
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import requests

# Make `bot.*` importable when Vercel runs this file from the repo root.
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
        print(f"[webhook] failed to send message: {exc!r}")


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # health check
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"TELEGRAM_BOT_TOKEN not configured")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            update = json.loads(raw.decode("utf-8") or "{}")
        except Exception as exc:  # noqa: BLE001
            print(f"[webhook] bad payload: {exc!r}")
            self.send_response(400)
            self.end_headers()
            return

        # Always 200 fast — Telegram retries on non-2xx and we don't want
        # duplicate sends if the LLM is slow. We process synchronously
        # within the function timeout (Vercel hobby = 10s, pro = 60s).
        try:
            replies = handle_update(update)
            for r in replies:
                _send(token, r)
        except Exception as exc:  # noqa: BLE001
            print(f"[webhook] handler crashed: {exc!r}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

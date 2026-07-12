"""Telegram notifications for the live bot.

Uses the plain Bot API over HTTPS (no extra dependency). Credentials come
from .env: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID. A notification failure
must NEVER break a trading tick — everything is best-effort and silent
when unconfigured.
"""

from __future__ import annotations

import json
import os
import urllib.request


def send(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return False
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps(
                {"chat_id": chat, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False

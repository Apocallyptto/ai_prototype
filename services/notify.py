# services/notify.py
from __future__ import annotations
import os
import json
import urllib.request
from typing import Optional, Dict, Any

# --- Env knobs (set only what you use) ---
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHATID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "")

TIMEOUT_SEC = 8


def _post_json(url: str, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # stdlib only; fail-soft upstream
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:  # noqa: S310
        _ = resp.read()


def _telegram_send(text: str) -> None:
    if not (TELEGRAM_TOKEN and TELEGRAM_CHATID):
        return  # no-op if not configured
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHATID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        _post_json(url, payload)
    except Exception:
        pass  # never raise from notifier


def _discord_send(text: str) -> None:
    if not DISCORD_WEBHOOK:
        return
    payload = {"content": text}
    try:
        _post_json(DISCORD_WEBHOOK, payload)
    except Exception:
        pass


def notify(text: str) -> None:
    """Broadcast a plain line to configured channels (if any)."""
    _telegram_send(text)
    _discord_send(text)


# ===== Convenience wrappers =====
def notify_info(text: str) -> None:
    notify(f"ℹ️ {text}")

def notify_warn(text: str) -> None:
    notify(f"⚠️ {text}")

def notify_error(text: str) -> None:
    notify(f"🛑 {text}")

def _fmt_price(x: Optional[float]) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)

def notify_trade_opened(symbol: str, side: str, qty: int,
                        entry: Optional[float],
                        tp: Optional[float],
                        sl: Optional[float],
                        reason: str = "") -> None:
    msg = (
        f"📈 <b>TRADE OPENED</b>\n"
        f"{symbol} {side} qty={qty}\n"
        f"entry={_fmt_price(entry)}  TP={_fmt_price(tp)}  SL={_fmt_price(sl)}"
    )
    if reason:
        msg += f"\nReason: {reason}"
    notify(msg)

def notify_trade_blocked(symbol: str, side: str, qty: int, reason: str) -> None:
    notify_warn(f"BLOCKED {symbol} {side} qty={qty}\n{reason}")

def notify_trade_skipped(symbol: str, side: str, why: str) -> None:
    notify_info(f"SKIP {symbol} {side}: {why}")

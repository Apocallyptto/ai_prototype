# services/notify.py
import os, json, logging, time
from typing import Optional, Dict, Any, Iterable
import requests

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("notify")

ENABLE = os.getenv("ENABLE_TELEGRAM", "0") == "1"
BOT = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

def _send_telegram(text: str) -> bool:
    if not ENABLE:
        return False
    if not BOT or not CHAT:
        log.warning("Telegram enabled but BOT/CHAT not set; skipping")
        return False
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT, "text": text, "parse_mode": "HTML"}, timeout=10)
        if r.status_code != 200:
            log.warning("Telegram non-200: %s %s", r.status_code, r.text[:200])
            return False
        return True
    except Exception as e:
        log.warning("Telegram send failed: %s", e)
        return False

def fmt_money(x: Optional[float]) -> str:
    if x is None: return "—"
    try: return f"${float(x):.2f}"
    except: return str(x)

def notify_signal(p: Dict[str, Any]) -> None:
    """
    p = {"symbol":"AAPL","side":"buy","strength":0.72,"px":268.45,"source":"ensemble"}
    """
    text = (
        f"🧠 <b>Signal</b> [{p.get('source','?')}]\n"
        f"• {p.get('symbol','?')}  • side: <b>{p.get('side','?').upper()}</b>\n"
        f"• strength: {p.get('strength','?')}  • px: {fmt_money(p.get('px'))}"
    )
    _send_telegram(text)

def notify_order_submitted(p: Dict[str, Any]) -> None:
    """
    p = {"symbol":"AAPL","side":"buy","qty":1,"limit":268.5,"tp":269.8,"sl":267.9,"id":"..."}
    """
    text = (
        f"📥 <b>Submitted</b> bracket\n"
        f"• {p.get('symbol')} {p.get('side','').upper()} qty {p.get('qty')}\n"
        f"• LMT {fmt_money(p.get('limit'))} | TP {fmt_money(p.get('tp'))} | SL {fmt_money(p.get('sl'))}\n"
        f"id: <code>{p.get('id','?')}</code>"
    )
    _send_telegram(text)

def notify_fill(p: Dict[str, Any]) -> None:
    """
    p = {"symbol":"AAPL","side":"buy","qty":1,"price":268.5,"avg":268.45,"order_id":"..."}
    """
    text = (
        f"✅ <b>Fill</b>\n"
        f"• {p.get('symbol')} {p.get('side','').upper()} qty {p.get('qty')} @ {fmt_money(p.get('price') or p.get('avg'))}\n"
        f"order: <code>{p.get('order_id','?')}</code>"
    )
    _send_telegram(text)

def notify_info(msg: str) -> None:
    _send_telegram(f"ℹ️ {msg}")

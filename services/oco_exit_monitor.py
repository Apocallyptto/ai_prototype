# services/oco_exit_monitor.py
from __future__ import annotations

import os
import time
import json
import logging
import datetime as dt
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ----------------------------
# Config
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "20"))

ATR_PCT = Decimal(os.getenv("ATR_PCT", "0.0100"))          # 1% default
TP_MULT = Decimal(os.getenv("TP_MULT", "1.75"))            # matches your observed TP/SL ~ 1.75 * ATR_PCT
SL_MULT = Decimal(os.getenv("SL_MULT", "1.75"))

PRICE_STEP = Decimal(os.getenv("PRICE_STEP", "0.01"))      # equities -> $0.01
MIN_REPLACE_INTERVAL_SEC = int(os.getenv("MIN_REPLACE_INTERVAL_SEC", "30"))

ORPHAN_MIN_AGE_SEC = int(os.getenv("ORPHAN_MIN_AGE_SEC", "120"))
CANCEL_ORPHAN_EXIT_ORDERS = os.getenv("CANCEL_ORPHAN_EXIT_ORDERS", "true").lower() in ("1", "true", "yes", "y")

DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes", "y")

CLIENT_PREFIX = os.getenv("EXIT_CLIENT_PREFIX", "EXIT-OCO")  # will create IDs like EXIT-OCO-AAPL-<epoch>


# Alpaca REST
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "").strip()   # e.g. https://paper-api.alpaca.markets

if not ALPACA_BASE_URL:
    # fallback: paper if not set
    mode = os.getenv("TRADING_MODE", "paper").lower()
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if mode == "paper" else "https://api.alpaca.markets"


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("oco_exit_monitor")


# ----------------------------
# Helpers
# ----------------------------
_TERMINAL_STATUSES = {"CANCELED", "FILLED", "REJECTED", "EXPIRED"}

def norm_status(st: Any) -> str:
    """
    Works for REST ('new') and alpaca-py ('OrderStatus.NEW') string-ish.
    """
    s = str(st).upper()
    s = s.split(".")[-1]
    return s

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def to_decimal(x: Any) -> Decimal:
    if x is None:
        return Decimal("0")
    return Decimal(str(x))

def round_to_step(px: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return px
    # quantize to step (e.g., 0.01)
    # convert to number of steps, round, multiply back
    steps = (px / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return (steps * step).quantize(step, rounding=ROUND_HALF_UP)

def client_id(symbol: str) -> str:
    return f"{CLIENT_PREFIX}-{symbol}-{int(time.time())}"

def is_exit_parent_order(o: Dict[str, Any], symbol: str) -> bool:
    cid = (o.get("client_order_id") or "")
    return cid.startswith(f"{CLIENT_PREFIX}-{symbol}-")

def is_active_order(o: Dict[str, Any]) -> bool:
    return norm_status(o.get("status")) not in _TERMINAL_STATUSES

def parse_created_at(o: Dict[str, Any]) -> Optional[dt.datetime]:
    # Alpaca REST returns ISO string like "2025-12-18T20:42:40.743238Z" or with +00:00
    s = o.get("created_at")
    if not s:
        return None
    try:
        # normalize Z
        s2 = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s2)
    except Exception:
        return None


# ----------------------------
# HTTP client (retries)
# ----------------------------
def make_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST", "DELETE"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

SESSION = make_session()

def alpaca_headers() -> Dict[str, str]:
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars.")
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def alpaca_url(path: str) -> str:
    return ALPACA_BASE_URL.rstrip("/") + path

def alpaca_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = SESSION.get(alpaca_url(path), headers=alpaca_headers(), params=params, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca GET {path} failed {r.status_code}: {r.text}")
    return r.json()

def alpaca_post(path: str, payload: Dict[str, Any]) -> Any:
    r = SESSION.post(alpaca_url(path), headers=alpaca_headers(), data=json.dumps(payload), timeout=20)
    if r.status_code >= 400:
        # bubble up error text to allow fallback logic
        raise RuntimeError(f"Alpaca POST {path} failed {r.status_code}: {r.text}")
    return r.json()

def alpaca_delete(path: str) -> Any:
    r = SESSION.delete(alpaca_url(path), headers=alpaca_headers(), timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"Alpaca DELETE {path} failed {r.status_code}: {r.text}")
    if r.text.strip():
        try:
            return r.json()
        except Exception:
            return r.text
    return None


# ----------------------------
# Alpaca wrappers
# ----------------------------
def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return alpaca_get(f"/v2/positions/{symbol}")
    except Exception:
        return None

def list_orders(status: str, symbols: List[str], limit: int = 200) -> List[Dict[str, Any]]:
    # Alpaca supports symbols as comma-separated list
    params = {
        "status": status,
        "limit": str(limit),
        "direction": "desc",
        "nested": "true",
        "symbols": ",".join(symbols),
    }
    return alpaca_get("/v2/orders", params=params)

def cancel_order(order_id: str) -> None:
    if DRY_RUN:
        log.warning("DRY_RUN cancel_order %s", order_id)
        return
    alpaca_delete(f"/v2/orders/{order_id}")

def cancel_active_exit_parents(symbol: str) -> int:
    """
    Cancels all non-terminal parent EXIT-OCO-* orders for symbol (by client_order_id prefix).
    """
    orders = list_orders(status="all", symbols=[symbol], limit=500)
    active = [o for o in orders if is_exit_parent_order(o, symbol) and is_active_order(o)]
    for o in active:
        oid = o.get("id")
        if oid:
            log.info("Canceling active EXIT parent | %s | status=%s | client=%s", oid, norm_status(o.get("status")), o.get("client_order_id"))
            try:
                cancel_order(oid)
            except Exception as e:
                log.warning("Cancel failed for %s: %s", oid, e)
    return len(active)

def find_open_exit_parent(symbol: str) -> Optional[Dict[str, Any]]:
    open_orders = list_orders(status="open", symbols=[symbol], limit=200)
    for o in open_orders:
        if is_exit_parent_order(o, symbol) and is_active_order(o):
            return o
    return None

def cancel_orphan_exit_if_no_position(symbol: str) -> int:
    """
    If no position exists, cancel EXIT-OCO parents older than ORPHAN_MIN_AGE_SEC.
    """
    if not CANCEL_ORPHAN_EXIT_ORDERS:
        return 0

    pos = get_position(symbol)
    if pos is not None:
        return 0

    orders = list_orders(status="open", symbols=[symbol], limit=200)
    now = utcnow()
    canceled = 0

    for o in orders:
        if not is_exit_parent_order(o, symbol):
            continue
        created = parse_created_at(o)
        age = None if not created else (now - created).total_seconds()
        if age is None or age >= ORPHAN_MIN_AGE_SEC:
            oid = o.get("id")
            if oid:
                log.info("Cancel orphan EXIT parent | %s | age_sec=%s | client=%s", oid, None if age is None else int(age), o.get("client_order_id"))
                try:
                    cancel_order(oid)
                    canceled += 1
                except Exception as e:
                    log.warning("Cancel orphan failed for %s: %s", oid, e)

    return canceled

def submit_exit_oco(symbol: str, qty: Decimal, exit_side: str, tp: Decimal, sl: Decimal) -> Dict[str, Any]:
    """
    Tries 2 payload styles:
      A) top-level limit_price + stop_loss (matches what you saw: parent LIMIT + sibling STOP)
      B) take_profit + stop_loss style
    """
    cid = client_id(symbol)

    # A) parent is the TP limit order, plus stop_loss
    payload_a = {
        "symbol": symbol,
        "qty": str(qty),
        "side": exit_side.lower(),
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": str(tp),
        "order_class": "oco",
        "stop_loss": {"stop_price": str(sl)},
        "client_order_id": cid,
    }

    # B) (fallback) explicit take_profit object
    payload_b = {
        "symbol": symbol,
        "qty": str(qty),
        "side": exit_side.lower(),
        "time_in_force": "gtc",
        "order_class": "oco",
        "take_profit": {"limit_price": str(tp)},
        "stop_loss": {"stop_price": str(sl)},
        "client_order_id": cid,
    }

    if DRY_RUN:
        log.warning("DRY_RUN submit_exit_oco A: %s", payload_a)
        return {"dry_run": True, "payload": payload_a}

    try:
        return alpaca_post("/v2/orders", payload_a)
    except Exception as e_a:
        log.warning("OCO submit payload A failed, trying B. Error: %s", e_a)
        return alpaca_post("/v2/orders", payload_b)


# ----------------------------
# Core logic
# ----------------------------
_last_replace_ts: Dict[str, float] = {}

def needs_replace(existing_parent: Optional[Dict[str, Any]], want_qty: Decimal, want_side: str) -> bool:
    """
    Replace if:
      - no existing exit parent
      - existing has different qty/side
    """
    if existing_parent is None:
        return True

    # qty can be string in REST
    ex_qty = to_decimal(existing_parent.get("qty"))
    ex_side = str(existing_parent.get("side") or "").lower()

    if ex_qty != want_qty:
        return True
    if ex_side != want_side.lower():
        return True

    return False

def compute_tp_sl(avg: Decimal, side: str) -> Tuple[Decimal, Decimal]:
    """
    side = 'long' or 'short'
    """
    if side.lower() == "long":
        tp = avg * (Decimal("1") + (ATR_PCT * TP_MULT))
        sl = avg * (Decimal("1") - (ATR_PCT * SL_MULT))
    else:
        tp = avg * (Decimal("1") - (ATR_PCT * TP_MULT))
        sl = avg * (Decimal("1") + (ATR_PCT * SL_MULT))

    tp = round_to_step(tp, PRICE_STEP)
    sl = round_to_step(sl, PRICE_STEP)

    return tp, sl

def main() -> None:
    log.info(
        "oco_exit_monitor starting | SYMBOLS=%s | POLL=%ss | ATR_PCT=%s | TP_MULT=%s | SL_MULT=%s | PRICE_STEP=%s | ORPHAN_MIN_AGE_SEC=%s | CANCEL_ORPHAN=%s | DRY_RUN=%s | BASE_URL=%s",
        SYMBOLS, POLL_SECONDS, ATR_PCT, TP_MULT, SL_MULT, PRICE_STEP, ORPHAN_MIN_AGE_SEC, CANCEL_ORPHAN_EXIT_ORDERS, DRY_RUN, ALPACA_BASE_URL,
    )

    while True:
        try:
            for sym in SYMBOLS:
                # 1) cancel orphan EXITs if no position
                cancel_orphan_exit_if_no_position(sym)

                pos = get_position(sym)
                if pos is None:
                    continue

                side = str(pos.get("side") or "").lower()  # 'long' or 'short'
                qty = to_decimal(pos.get("qty"))
                avg = to_decimal(pos.get("avg_entry_price") or pos.get("avg_entry_price") or pos.get("avg_entry_price"))

                if qty <= 0 or avg <= 0 or side not in ("long", "short"):
                    log.warning("Skip %s invalid position | side=%s qty=%s avg=%s", sym, side, qty, avg)
                    continue

                exit_side = "sell" if side == "long" else "buy"

                # 2) ensure we have correct exit OCO
                parent = find_open_exit_parent(sym)
                if not needs_replace(parent, qty, exit_side):
                    # already good
                    continue

                # rate limit replace per symbol
                now_ts = time.time()
                last = _last_replace_ts.get(sym, 0.0)
                if (now_ts - last) < MIN_REPLACE_INTERVAL_SEC:
                    continue

                # cancel previous active parents (and Alpaca will cancel linked leg)
                canceled = cancel_active_exit_parents(sym)

                tp, sl = compute_tp_sl(avg, side)

                # sanity checks
                if side == "long" and not (tp > avg and sl < avg):
                    log.warning("Computed weird TP/SL for LONG %s | avg=%s tp=%s sl=%s (skip)", sym, avg, tp, sl)
                    continue
                if side == "short" and not (tp < avg and sl > avg):
                    log.warning("Computed weird TP/SL for SHORT %s | avg=%s tp=%s sl=%s (skip)", sym, avg, tp, sl)
                    continue

                resp = submit_exit_oco(sym, qty, exit_side, tp, sl)

                oid = resp.get("id") if isinstance(resp, dict) else None
                log.info(
                    "Placed EXIT-OCO | %s qty=%s side=%s avg=%s -> TP=%s SL=%s | order_id=%s | canceled_before=%s",
                    sym, qty, side, avg, tp, sl, oid, canceled
                )

                _last_replace_ts[sym] = now_ts

        except Exception as e:
            log.exception("oco_exit_monitor loop error: %s", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

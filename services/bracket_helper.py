# services/executor_bracket.py
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

# DB drivers: prefer psycopg3, fall back to psycopg2
try:
    import psycopg  # psycopg3
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2 as psycopg

from services.bracket_helper import submit_bracket

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
if API_KEY and API_SECRET:
    S.headers.update({
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
    })

PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
WINDOW_MIN   = int(os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN", "15"))  # <-- new
WASH_LOCK_MIN = int(os.getenv("WASH_LOCK_MIN", "5"))
MAX_PARENTS_PER_SYMBOL = int(os.getenv("MAX_PARENTS_PER_SYMBOL", "2"))  # cap open parents per symbol

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set.")
    sys.exit(1)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _db_connect():
    if HAVE3:
        return psycopg.connect(DATABASE_URL)
    return psycopg.connect(DATABASE_URL)

def _fetch_latest_signals(window_min: int, min_strength: float, pid: int) -> Dict[str, Dict]:
    """
    Get the newest signal for each symbol within the last `window_min` minutes,
    filtered by `min_strength` and portfolio_id.
    Returns: { "AAPL": {"side": "buy", "strength": 0.62, "created_at": "..."} , ... }
    """
    since = _utcnow() - timedelta(minutes=window_min)
    out: Dict[str, Dict] = {}

    sql = """
        SELECT s.symbol, s.side, s.strength, s.created_at
        FROM public.signals s
        JOIN (
            SELECT symbol, MAX(created_at) AS mx
            FROM public.signals
            WHERE portfolio_id = %s AND created_at >= %s AND strength >= %s
            GROUP BY symbol
        ) t
        ON s.symbol = t.symbol AND s.created_at = t.mx
        WHERE s.portfolio_id = %s
        ORDER BY s.created_at DESC;
    """

    with _db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (pid, since, float(min_strength), pid))
            for sym, side, strength, created_at in cur.fetchall():
                sym = sym.upper()
                if SYMBOLS and sym not in SYMBOLS:
                    continue
                out[sym] = {
                    "side": side,
                    "strength": float(strength),
                    "created_at": created_at,
                }
    return out

def _alpaca_open_parent_count(symbol: str) -> int:
    """Count open parent bracket orders at Alpaca for this symbol."""
    r = S.get(f"{ALPACA_BASE_URL}/v2/orders", params={"status": "open", "symbols": symbol, "limit": 200}, timeout=20)
    r.raise_for_status()
    orders = r.json()
    # Count parents (order_class == 'bracket' and not child legs)
    parents = 0
    for o in orders:
        if (o.get("order_class") == "bracket") and (o.get("symbol","").upper() == symbol.upper()):
            # Parent has no 'parent_order_id'
            if not o.get("parent_order_id"):
                parents += 1
    return parents

# simple in-memory wash-lock
_LAST_SIDE_TIME: Dict[Tuple[str, str], float] = {}

def _wash_locked(symbol: str, side: str, now_ts: float) -> Optional[str]:
    key = (symbol, side)
    last = _LAST_SIDE_TIME.get(key)
    if last is None:
        return None
    remain = WASH_LOCK_MIN*60 - (now_ts - last)
    if remain > 0:
        mins = int(remain // 60)
        secs = int(remain % 60)
        return f"{mins}m {secs}s"
    return None

def _note_submit(symbol: str, side: str, now_ts: float):
    _LAST_SIDE_TIME[(symbol, side)] = now_ts

def place_from_signals():
    sigs = _fetch_latest_signals(WINDOW_MIN, MIN_STRENGTH, PORTFOLIO_ID)
    if not sigs:
        print("INFO executor_bracket | No signals within window.")
        return

    placed = 0
    now_ts = time.time()

    for sym in SYMBOLS:
        sig = sigs.get(sym)
        if not sig:
            continue

        side = sig["side"].lower().strip()
        strength = float(sig["strength"])

        # wash-trade lock
        msg = _wash_locked(sym, side, now_ts)
        if msg:
            print(f"WARN executor_bracket | {sym} {side}: wash-trade lock active; {msg} remaining. Skipping.")
            continue

        # parent cap
        try:
            parents = _alpaca_open_parent_count(sym)
            if parents >= MAX_PARENTS_PER_SYMBOL:
                print(f"INFO executor_bracket | {sym} {side}: open-parent cap reached ({parents}); skipping.")
                continue
        except Exception as e:
            print(f"WARN executor_bracket | {sym}: open-parent count failed -> {e}. Continuing cautiously.")

        # client id
        client_id = f"BRK-{sym}-{int(now_ts)}"

        try:
            print(f"INFO executor_bracket | Submit bracket for {sym} {side} (strength={strength:.2f})")
            # qty is computed dynamically inside submit_bracket if USE_DYNAMIC_SIZE=1
            submit_bracket(symbol=sym, side=side, strength=strength, qty=None, client_id=client_id,
                           time_in_force="day", order_type="market", extended_hours=False)
            placed += 1
            _note_submit(sym, side, now_ts)
        except requests.HTTPError as he:
            # show Alpaca error payload clearly
            try:
                print(f"ERROR executor_bracket | {sym} {side}: submit failed -> {he.response.text}")
            except Exception:
                print(f"ERROR executor_bracket | {sym} {side}: submit failed -> {he}")
        except Exception as e:
            print(f"ERROR executor_bracket | {sym} {side}: submit failed -> {e}")

    print(f"INFO executor_bracket | Done. Placed {placed} bracket order(s).")

def main():
    place_from_signals()

if __name__ == "__main__":
    main()

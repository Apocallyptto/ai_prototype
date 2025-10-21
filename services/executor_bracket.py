# services/executor_bracket.py
"""
Executor: read recent signals from DB and submit bracket entries to Alpaca,
with hard guards to prevent duplicate spam.

Guards:
- Wash-trade cooldown: skip if a parent bracket for (symbol, side) was created
  within the last WASH_LOCK_MIN minutes (checked via Alpaca order history).
- Open-order guard: skip if there is already an OPEN parent bracket (no parent_order_id)
  for the same (symbol, side).

Env:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname
  PORTFOLIO_ID=1
  MIN_STRENGTH=0.45
  WASH_LOCK_MIN=30      # minutes
  DEFAULT_QTY=1
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
  ALPACA_API_KEY=...
  ALPACA_API_SECRET=...

CLI:
  $env:PYTHONPATH="$PWD"
  python -m services.executor_bracket --since-days 1 --min-strength 0.45
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import shlex
import psutil  # optional, but harmless if present
from typing import List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras
import requests

# --- Config / Env ---
DATABASE_URL = os.getenv("DATABASE_URL")
PORTFOLIO_ID = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH_ENV = float(os.getenv("MIN_STRENGTH", "0.45"))
WASH_LOCK_MIN = int(os.getenv("WASH_LOCK_MIN", "30"))
DEFAULT_QTY = os.getenv("DEFAULT_QTY", "1")

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")

# bracket helper – we only need the submit function
try:
    # Prefer the more explicit name if present
    from services.bracket_helper import submit_bracket_entry as _submit_bracket
except ImportError:
    # Fall back to generic name used elsewhere in your repo
    from services.bracket_helper import submit_bracket as _submit_bracket  # type: ignore

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    "Content-Type": "application/json",
})

def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"{ts} {level} executor_bracket | {msg}")

# --- DB helpers ---
def _pg_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)

def fetch_signals(since_days: int, min_strength: float, portfolio_id: int) -> List[dict]:
    """
    Return latest signal per symbol meeting the min strength in the time window.
    Table schema expected: signals(symbol TEXT, side TEXT, strength REAL, created_at TIMESTAMPTZ, portfolio_id INT)
    """
    q = """
    with ranked as (
        select
            symbol,
            side,
            strength,
            created_at,
            portfolio_id,
            row_number() over (partition by symbol order by created_at desc) as rn
        from signals
        where created_at >= now() - interval %s
          and strength >= %s
          and portfolio_id = %s
    )
    select symbol, side, strength, created_at
    from ranked
    where rn = 1
    order by symbol asc;
    """
    interval = f"{max(1, since_days)} days"
    with _pg_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(q, (interval, float(min_strength), int(portfolio_id)))
        rows = cur.fetchall()
        return [dict(r) for r in rows]

# --- Alpaca helpers ---
def _http(method: str, url: str, **kwargs) -> requests.Response:
    # modest retry on 5xx
    for attempt in range(4):
        r = SESSION.request(method, url, timeout=15, **kwargs)
        if r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        return r
    return r  # last response

def get_open_parent_brackets(symbol: str, side: str) -> List[dict]:
    """
    Returns open PARENT bracket orders (no parent_order_id) for symbol & side.
    """
    url = f"{ALPACA_BASE_URL}/v2/orders?status=open&nested=true&symbols={symbol}"
    r = _http("GET", url)
    r.raise_for_status()
    out = []
    for o in r.json():
        if o.get("symbol") != symbol:
            continue
        if o.get("order_class") != "bracket":
            continue
        if o.get("side") != side:
            continue
        if o.get("parent_order_id"):   # only parent
            continue
        out.append(o)
    return out

def last_parent_bracket_time(symbol: str, side: str) -> Optional[datetime]:
    """
    Look back through recent orders and return the most recent parent bracket created_at for (symbol, side).
    """
    url = f"{ALPACA_BASE_URL}/v2/orders?status=all&nested=false&symbols={symbol}&limit=100"
    r = _http("GET", url)
    if r.status_code == 403:
        # If permissions are limited, just return None; cooldown will effectively be disabled.
        log(f"403 when reading order history for {symbol} — cannot enforce wash lock from history.", "WARN")
        return None
    r.raise_for_status()
    orders = r.json()
    latest: Optional[datetime] = None
    for o in orders:
        if o.get("symbol") != symbol:
            continue
        if o.get("order_class") != "bracket":
            continue
        if o.get("side") != side:
            continue
        if o.get("parent_order_id"):  # parent only
            continue
        ts = o.get("created_at")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest

def within_wash_lock(symbol: str, side: str, minutes: int) -> bool:
    last = last_parent_bracket_time(symbol, side)
    if not last:
        return False
    now = datetime.now(timezone.utc)
    return (now - last) < timedelta(minutes=minutes)

# --- Core execution ---
def place_from_signals(since_days: int, min_strength: float):
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing in env")

    signals = fetch_signals(since_days, min_strength, PORTFOLIO_ID)
    if not signals:
        log("No signals within dedupe window.")
        return

    placed = 0
    for sig in signals:
        sym = sig["symbol"].upper()
        side = sig["side"].lower()
        strength = float(sig["strength"])

        # only long entries are supported in this executor (extend if you trade short)
        if side not in ("buy", "sell"):
            log(f"{sym} {side}: unsupported side; skipping.", "WARN")
            continue

        # Open-order guard
        try:
            open_parents = get_open_parent_brackets(sym, side)
        except Exception as e:
            log(f"{sym} {side}: failed to query open orders -> {e}", "WARN")
            open_parents = []
        if open_parents:
            log(f"{sym} {side}: open parent bracket already exists; skipping.")
            continue

        # Wash-trade cooldown
        try:
            if within_wash_lock(sym, side, WASH_LOCK_MIN):
                log(f"{sym} {side}: wash-trade lock active; skipping.", "WARN")
                continue
        except Exception as e:
            log(f"{sym} {side}: could not evaluate wash lock ({e}); continuing cautiously.", "WARN")

        # Submit via bracket_helper (which handles clock/ATR/limit-vs-market choice)
        qty = str(DEFAULT_QTY)
        try:
            log(f"Submit bracket for {sym} {side} (strength={strength:.2f})")
            _submit_bracket(symbol=sym, side=side, qty=qty)
            placed += 1
        except Exception as e:
            log(f"{sym} {side}: submit failed -> {e}", "ERROR")

    log(f"Done. Placed {placed} bracket order(s).")

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit bracket entries from recent signals with guards.")
    ap.add_argument("--since-days", type=int, default=1, help="How many days of signals to look back (default 1)")
    ap.add_argument("--min-strength", type=float, default=MIN_STRENGTH_ENV, help="Min strength gate")
    return ap.parse_args(argv)

if __name__ == "__main__":
    args = parse_cli(sys.argv[1:])
    place_from_signals(args.since_days, args.min_strength)

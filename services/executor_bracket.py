"""
services/executor_bracket.py

Places ATR-aware bracket parent orders from recent signals in the DB.
- Reads 'signals' table and filters by time window & min strength.
- Wash-trade lock to avoid immediate re-entries.
- Caps concurrent open parents per symbol and daily new parents.
- Delegates actual order submission to services.bracket_helper.submit_bracket.

Usage:
    $env:PYTHONPATH = "$PWD"
    python -m services.executor_bracket --since-days 1 --min-strength 0.55

Env (optional):
    ALPACA_BASE_URL        (default https://paper-api.alpaca.markets)
    ALPACA_API_KEY, ALPACA_API_SECRET
    ALPACA_DATA_FEED       (iex/sip) – only used by bracket_helper

    DATABASE_URL or PGHOST/PGUSER/PGPASSWORD/PGDATABASE/PGPORT

    SYMBOLS                (comma list; default: AAPL,MSFT,SPY)
    MIN_STRENGTH           (float; CLI flag has priority)
    MAX_POSITIONS          (int; currently informational)
    PORTFOLIO_ID           (int; default 1)
    QTY_PER_TRADE          (int; default 1)

    WASH_LOCK_MIN          (int; default 20) — min minutes since last parent for (symbol, side)
    MAX_PARENTS_PER_SYMBOL (int; default 1)  — cap on concurrently OPEN parents for (symbol, side)
    DAILY_PARENTS_PER_SYMBOL (int; default 5) — cap on PARENTS created today for (symbol, side)
"""

from __future__ import annotations

import os
import sys
import json
import time
import shlex
import psutil as _psutil  # will be made optional below
from typing import List, Tuple, Optional, Iterable, Dict
from datetime import datetime, timedelta, timezone

# Make psutil truly optional
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # fallback

import requests
import psycopg2
from psycopg2.extras import RealDictCursor

# Reuse your submission logic (no duplication)
from services.bracket_helper import submit_bracket

# -------------------- Config -------------------- #

ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY           = os.getenv("ALPACA_API_KEY", "")
API_SECRET        = os.getenv("ALPACA_API_SECRET", "")

SYMBOLS           = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
DEFAULT_MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.55"))
PORTFOLIO_ID      = int(os.getenv("PORTFOLIO_ID", "1"))
QTY_PER_TRADE     = int(os.getenv("QTY_PER_TRADE", "1"))

WASH_LOCK_MIN     = int(os.getenv("WASH_LOCK_MIN", "20"))
MAX_PARENTS_PER_SYMBOL   = int(os.getenv("MAX_PARENTS_PER_SYMBOL", "1"))
DAILY_PARENTS_PER_SYMBOL = int(os.getenv("DAILY_PARENTS_PER_SYMBOL", "5"))

LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO").upper()

# HTTP session with Alpaca creds
SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
})

# -------------------- Logging -------------------- #

def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG", "INFO", "WARN", "ERROR"]
    if level not in levels:
        level = "INFO"
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} executor_bracket | {msg}")

# -------------------- DB Helpers -------------------- #

def _pg_conn():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    host = os.getenv("PGHOST", "localhost")
    user = os.getenv("PGUSER", "postgres")
    pwd  = os.getenv("PGPASSWORD", "postgres")
    db   = os.getenv("PGDATABASE", "ai_prototype")
    port = int(os.getenv("PGPORT", "5432"))
    return psycopg2.connect(host=host, user=user, password=pwd, dbname=db, port=port)

def fetch_signals(since_days: int, min_strength: float, portfolio_id: Optional[int]):
    since = datetime.now(timezone.utc) - timedelta(days=since_days)
    sql = """
        SELECT DISTINCT ON (symbol, side)
               symbol,
               side,
               strength,
               created_at AS ts
        FROM signals
        WHERE created_at >= %s
          AND strength   >= %s
          AND (portfolio_id = COALESCE(%s, portfolio_id))
        ORDER BY symbol, side, created_at DESC
        LIMIT 1000
    """
    with psycopg2.connect(dsn=os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(sql, (since, float(min_strength), None if portfolio_id in (None, "", "null") else int(portfolio_id)))
        return cur.fetchall()


# -------------------- Alpaca order lookups -------------------- #

def _http(method: str, url: str, **kwargs) -> requests.Response:
    # light retries
    for attempt in range(4):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            wait = min(2 ** attempt, 8)
            log(f"HTTP {method} {url} failed: {e} -> retrying in {wait}s", "WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

def get_open_parent_brackets(symbol: str, side: str) -> List[dict]:
    """
    Returns open parent bracket orders (no parent_order_id) for (symbol, side).
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {
        "status": "open",
        "symbols": symbol,
        "nested": "false",
        "limit": 200,
    }
    r = _http("GET", url, params=params)
    r.raise_for_status()
    out = []
    for o in r.json():
        if o.get("symbol") != symbol:
            continue
        if o.get("order_class") != "bracket":
            continue
        if o.get("parent_order_id"):  # only parents
            continue
        if o.get("side") != side:
            continue
        out.append(o)
    return out

def count_open_parent_brackets(symbol: str, side: str) -> int:
    try:
        return len(get_open_parent_brackets(symbol, side))
    except Exception as e:
        log(f"{symbol} {side}: open-parent check failed: {e}", "WARN")
        return 0  # be permissive on failure

def last_parent_bracket_time(symbol: str, side: str) -> Optional[datetime]:
    """
    Latest created_at for any PARENT bracket for (symbol, side) with any status.
    """
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {
        "status": "all",
        "symbols": symbol,
        "nested": "false",
        "limit": 200,
    }
    r = _http("GET", url, params=params)
    r.raise_for_status()
    last: Optional[datetime] = None
    for o in r.json():
        if o.get("symbol") != symbol:
            continue
        if o.get("order_class") != "bracket":
            continue
        if o.get("parent_order_id"):
            continue
        if o.get("side") != side:
            continue
        ts = o.get("created_at")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            continue
        if (last is None) or (t > last):
            last = t
    return last

def wash_lock_remaining(symbol: str, side: str, minutes: int) -> Optional[timedelta]:
    last = last_parent_bracket_time(symbol, side)
    if not last:
        return None
    now = datetime.now(timezone.utc)
    rem = timedelta(minutes=minutes) - (now - last)
    return rem if rem > timedelta(0) else None

def count_parents_today(symbol: str, side: str) -> int:
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {
        "status": "all",
        "symbols": symbol,
        "nested": "false",
        "limit": 200,
    }
    r = _http("GET", url, params=params)
    r.raise_for_status()
    today = datetime.now(timezone.utc).date()
    cnt = 0
    for o in r.json():
        if o.get("symbol") != symbol:
            continue
        if o.get("order_class") != "bracket":
            continue
        if o.get("parent_order_id"):
            continue
        if o.get("side") != side:
            continue
        ts = o.get("created_at")
        if not ts:
            continue
        try:
            d = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
        except Exception:
            continue
        if d == today:
            cnt += 1
    return cnt

# -------------------- Core flow -------------------- #

def place_from_signals(since_days: int, min_strength: float, portfolio_id: int, max_positions: Optional[int] = None):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    sigs = fetch_signals(since_days, min_strength, portfolio_id)
    if not sigs:
        log("No signals within dedupe window.")
        return

    # Optional: enforce symbol universe filter
    universe = set(SYMBOLS) if SYMBOLS else None

    placed = 0
    for symbol, side, strength, ts in sigs:
        if universe and symbol not in universe:
            continue

        # Wash-lock
        rem = wash_lock_remaining(symbol, side, WASH_LOCK_MIN)
        if rem:
            m = int(rem.total_seconds() // 60)
            s = int(rem.total_seconds() % 60)
            log(f"{symbol} {side}: wash-trade lock active; {m}m {s}s remaining. Skipping.", "WARN")
            continue

        # Cap concurrent open parents
        open_parents = count_open_parent_brackets(symbol, side)
        if open_parents >= MAX_PARENTS_PER_SYMBOL:
            log(f"{symbol} {side}: open-parent cap reached ({MAX_PARENTS_PER_SYMBOL}); skipping.")
            continue

        # Cap daily parents
        try:
            parents_today = count_parents_today(symbol, side)
        except Exception as e:
            log(f"{symbol} {side}: daily parent count failed ({e}); treating as 0.", "WARN")
            parents_today = 0

        if parents_today >= DAILY_PARENTS_PER_SYMBOL:
            log(f"{symbol} {side}: daily parent cap reached ({DAILY_PARENTS_PER_SYMBOL}); skipping.")
            continue

        # If a parent bracket is already open (rare race with previous check), skip
        if open_parents > 0:
            log(f"{symbol} {side}: open parent bracket already exists; skipping.")
            continue

        # Submit via your helper (keeps ATR/limit/market logic in one place)
        try:
            log(f"Submit bracket for {symbol} {side} (strength={strength:.2f})")
            _ = submit_bracket(symbol=symbol, side=side, qty=str(QTY_PER_TRADE))
            placed += 1
        except requests.HTTPError as he:
            # surface Alpaca message
            try:
                msg = he.response.text
            except Exception:
                msg = str(he)
            log(f"{symbol} {side}: submit failed -> {msg}", "ERROR")
        except Exception as e:
            log(f"{symbol} {side}: submit failed -> {e}", "ERROR")

    log(f"Done. Placed {placed} bracket order(s).")

# -------------------- CLI -------------------- #

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit bracket parent orders from DB signals.")
    ap.add_argument("--since-days", type=int, default=1, help="Lookback for signals (days).")
    ap.add_argument("--min-strength", type=float, default=DEFAULT_MIN_STRENGTH, help="Minimum signal strength.")
    ap.add_argument("--portfolio-id", type=int, default=PORTFOLIO_ID, help="Portfolio id in signals table.")
    ap.add_argument("--max-positions", type=int, default=int(os.getenv("MAX_POSITIONS", "10")),
                    help="Reserved for future use.")
    return ap.parse_args(argv)

def main():
    args = parse_cli(sys.argv[1:])
    place_from_signals(args.since_days, args.min_strength, args.portfolio_id, args.max_positions)

if __name__ == "__main__":
    main()

"""
services/executor_bracket.py

Places ATR-aware bracket parent orders from recent signals in the DB.
- Reads 'public.signals' and filters by time window & min strength.
- Wash-trade lock to avoid immediate re-entries.
- Caps concurrent open parents per symbol and daily new parents.
- Skips opposite-side parents and conflicts with live positions.
- Delegates submit to services.bracket_helper.submit_bracket.

Usage:
    $env:PYTHONPATH = "$PWD"
    python -m services.executor_bracket --since-days 1 --min-strength 0.55

Env:
    ALPACA_BASE_URL, ALPACA_API_KEY, ALPACA_API_SECRET
    ALPACA_DATA_FEED            (iex/sip) – used by bracket_helper
    DATABASE_URL or PG*         (PGHOST/PGUSER/PGPASSWORD/PGDATABASE/PGPORT, PGSSLMODE)

    SYMBOLS                     (default: AAPL,MSFT,SPY)
    MIN_STRENGTH                (float; CLI wins)
    PORTFOLIO_ID                (default 1)
    QTY_PER_TRADE               (default 1)

    WASH_LOCK_MIN               (default 20)  min minutes between parent re-entries (symbol,side)
    MAX_PARENTS_PER_SYMBOL      (default 1)   cap on concurrently OPEN parents (symbol,side)
    DAILY_PARENTS_PER_SYMBOL    (default 5)   cap on parents created today (symbol,side)
"""
from __future__ import annotations

import os, sys, json, time
from typing import List, Optional
from datetime import datetime, timedelta, timezone

import requests
import psycopg2
from services.bracket_helper import submit_bracket  # keeps ATR/ETH/RTH logic centralized

# -------------------- Config -------------------- #

ALPACA_BASE_URL   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY           = os.getenv("ALPACA_API_KEY", "")
API_SECRET        = os.getenv("ALPACA_API_SECRET", "")

SYMBOLS                 = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
DEFAULT_MIN_STRENGTH    = float(os.getenv("MIN_STRENGTH", "0.55"))
PORTFOLIO_ID            = int(os.getenv("PORTFOLIO_ID", "1"))
QTY_PER_TRADE           = int(os.getenv("QTY_PER_TRADE", "1"))

WASH_LOCK_MIN           = int(os.getenv("WASH_LOCK_MIN", "20"))
MAX_PARENTS_PER_SYMBOL  = int(os.getenv("MAX_PARENTS_PER_SYMBOL", "1"))
DAILY_PARENTS_PER_SYMBOL= int(os.getenv("DAILY_PARENTS_PER_SYMBOL", "5"))

LOG_LEVEL               = os.getenv("LOG_LEVEL", "INFO").upper()

SESSION = requests.Session()
SESSION.headers.update({
    "APCA-API-KEY-ID": API_KEY or "",
    "APCA-API-SECRET-KEY": API_SECRET or "",
    "Content-Type": "application/json",
})

# -------------------- Logging -------------------- #

def log(msg: str, level: str = "INFO"):
    order = ["DEBUG","INFO","WARN","ERROR"]
    if level not in order:
        level = "INFO"
    if order.index(level) >= order.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} executor_bracket | {msg}", flush=True)

def _http(method: str, url: str, **kwargs) -> requests.Response:
    for attempt in range(4):
        try:
            r = SESSION.request(method, url, timeout=15, **kwargs)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            wait = min(2**attempt, 8)
            log(f"HTTP {method} {url} failed: {e} -> retry {wait}s", "WARN")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed after retries: {method} {url}")

# -------------------- DB Helpers -------------------- #

def _pg_conn():
    """Prefer DATABASE_URL, else PG* envs, else localhost."""
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
    """
    Pull the most recent signal per (symbol, side) since a given horizon,
    using created_at if present else ts, and optionally filter by portfolio_id.
    Based on your earlier version.  (Ref: current file content)  :contentReference[oaicite:3]{index=3}
    """
    since = datetime.now(timezone.utc) - timedelta(days=since_days)
    sql = """
        SELECT DISTINCT ON (symbol, side)
               symbol,
               side,
               strength,
               COALESCE(created_at, ts) AS ts
        FROM public.signals
        WHERE COALESCE(created_at, ts) >= %s
          AND strength >= %s
          AND (%s IS NULL OR portfolio_id = %s)
        ORDER BY symbol, side, COALESCE(created_at, ts) DESC
        LIMIT 1000
    """
    with _pg_conn() as conn, conn.cursor() as cur:
        pid = None if portfolio_id in (None, "", "null") else int(portfolio_id)
        cur.execute(sql, (since, float(min_strength), pid, pid))
        return cur.fetchall()

# -------------------- Alpaca lookups -------------------- #

def get_open_parent_brackets(symbol: str, side: str) -> list[dict]:
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {"status":"open", "symbols":symbol, "nested":"false", "limit":200}
    r = _http("GET", url, params=params); r.raise_for_status()
    out = []
    for o in r.json():
        if o.get("symbol") != symbol: continue
        if o.get("order_class") != "bracket": continue
        if o.get("parent_order_id"): continue  # parents only
        if o.get("side") != side: continue
        out.append(o)
    return out

def count_open_parent_brackets(symbol: str, side: str) -> int:
    try:
        return len(get_open_parent_brackets(symbol, side))
    except Exception as e:
        log(f"{symbol} {side}: open-parent check failed: {e}", "WARN")
        return 0

def last_parent_bracket_time(symbol: str, side: str) -> Optional[datetime]:
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {"status":"all","symbols":symbol,"nested":"false","limit":200}
    r = _http("GET", url, params=params); r.raise_for_status()
    last: Optional[datetime] = None
    for o in r.json():
        if o.get("symbol")!=symbol or o.get("order_class")!="bracket" or o.get("parent_order_id") or o.get("side")!=side:
            continue
        ts = o.get("created_at")
        if not ts: continue
        try:
            t = datetime.fromisoformat(str(ts).replace("Z","+00:00"))
        except Exception:
            continue
        if last is None or t > last:
            last = t
    return last

def wash_lock_remaining(symbol: str, side: str, minutes: int) -> Optional[timedelta]:
    last = last_parent_bracket_time(symbol, side)
    if not last: return None
    rem = timedelta(minutes=minutes) - (datetime.now(timezone.utc) - last)
    return rem if rem > timedelta(0) else None

def count_parents_today(symbol: str, side: str) -> int:
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {"status":"all","symbols":symbol,"nested":"false","limit":200}
    r = _http("GET", url, params=params); r.raise_for_status()
    today = datetime.now(timezone.utc).date()
    c = 0
    for o in r.json():
        if o.get("symbol")!=symbol or o.get("order_class")!="bracket" or o.get("parent_order_id") or o.get("side")!=side:
            continue
        ts = o.get("created_at")
        if not ts: continue
        try:
            d = datetime.fromisoformat(str(ts).replace("Z","+00:00")).date()
        except Exception:
            continue
        if d == today: c += 1
    return c

def has_conflicting_position(symbol: str, side: str) -> bool:
    """True if we hold a live position on the opposite side (prevents Alpaca 403)."""
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    r = SESSION.get(url, timeout=10, headers=SESSION.headers)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    js = r.json()
    qty = float(js.get("qty", "0") or 0)
    # long qty>0 conflicts with selling short; short qty<0 conflicts with buying to open
    return (side == "sell" and qty > 0) or (side == "buy" and qty < 0)

# -------------------- Core flow -------------------- #

def place_from_signals(since_days: int, min_strength: float, portfolio_id: int, max_positions: Optional[int] = None):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    sigs = fetch_signals(since_days, min_strength, portfolio_id)
    if not sigs:
        log("No signals within window.")
        return

    universe = set(SYMBOLS) if SYMBOLS else None
    placed = 0

    for symbol, side, strength, ts in sigs:
        if universe and symbol not in universe:
            continue

        # Wash-lock
        rem = wash_lock_remaining(symbol, side, WASH_LOCK_MIN)
        if rem:
            m, s = int(rem.total_seconds()//60), int(rem.total_seconds()%60)
            log(f"{symbol} {side}: wash-trade lock active; {m}m {s}s remaining. Skipping.", "WARN")
            continue

        # Cap open parents (same side)
        open_same = count_open_parent_brackets(symbol, side)
        if open_same >= MAX_PARENTS_PER_SYMBOL:
            log(f"{symbol} {side}: open-parent cap reached ({MAX_PARENTS_PER_SYMBOL}); skipping.")
            continue

        # Daily parent cap (same side)
        try:
            parents_today = count_parents_today(symbol, side)
        except Exception as e:
            log(f"{symbol} {side}: daily parent count failed ({e}); treating as 0.", "WARN")
            parents_today = 0
        if parents_today >= DAILY_PARENTS_PER_SYMBOL:
            log(f"{symbol} {side}: daily parent cap reached ({DAILY_PARENTS_PER_SYMBOL}); skipping.")
            continue

        # Opposite-side open parent guard (preempt Alpaca 403)
        opp = "sell" if side == "buy" else "buy"
        open_opp = count_open_parent_brackets(symbol, opp)
        if open_opp > 0:
            log(f"{symbol} {side}: opposite-side parent open ({opp}); skipping.", "WARN")
            continue

        # Live position conflict guard (avoid short-against-long / long-against-short)
        try:
            if has_meat := has_conflicting_position(symbol, side):
                log(f"{symbol} {side}: conflicting live position; skipping.", "WARN")
                continue
        except Exception as e:
            # Be conservative
            log(f"{symbol} {side}: position check failed ({e}); skipping.", "WARN")
            continue

        # Submit
        try:
            log(f"Submit bracket for {symbol} {side} (strength={strength:.2f})")
            _ = submit_bracket(symbol=symbol, side=side, qty=str(QTY_PER_TRADE))
            placed += 1
        except requests.HTTPError as he:
            msg = getattr(he, "response", None)
            msg = msg.text if msg is not None else str(he)
            log(f"{symbol} {side}: submit failed -> {msg}", "ERROR")
        except Exception as e:
            log(f"{symbol} {side}: submit failed -> {e}", "ERROR")

    log(f"Done. Placed {placed} bracket order(s).")

# -------------------- CLI -------------------- #

def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit bracket parent orders from DB signals.")
    ap.add_argument("--since-days", type=int, default=1)
    ap.add_argument("--min-strength", type=float, default=DEFAULT_MIN_STRENGTH)
    ap.add_argument("--portfolio-id", type=int, default=PORTFOLIO_ID)
    ap.add_argument("--max-positions", type=int, default=int(os.getenv("MAX_POSITIONS","10")))
    return ap.parse_args(argv)

def main():
    args = parse_cli(sys.argv[1:])
    place_from_signals(args.since_days, args.min_strength, args.portfolio_id, args.max_positions)

if __name__ == "__main__":
    main()

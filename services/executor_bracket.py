# services/executor_bracket.py
from __future__ import annotations
import os, sys, time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Dict

# DB
import json
import math

# Prefer psycopg (psycopg3); fallback to psycopg2 if needed
try:
    import psycopg  # type: ignore
    HAVE_PSYCOPG3 = True
except Exception:
    HAVE_PSYCOPG3 = False
    try:
        import psycopg2  # type: ignore
    except Exception as e:
        raise RuntimeError("Install psycopg or psycopg2 to use executor_bracket.py") from e

import requests

# Reuse the bracket entry helper
from services.bracket_helper import submit_bracket_entry, list_open_orders

# ---------------- Env & knobs ---------------- #

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgres://user:pass@host:5432/db
PGHOST = os.getenv("PGHOST")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGDATABASE = os.getenv("PGDATABASE")
PGPORT = os.getenv("PGPORT", "5432")

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

PORTFOLIO_ID   = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH   = float(os.getenv("MIN_STRENGTH", "0.30"))
SINCE_DAYS     = int(os.getenv("SINCE_DAYS", "1"))
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", "10"))
DEDUPE_MINS    = int(os.getenv("DEDUPE_MINS", "10"))  # 10-min dedupe window
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO").upper()

# Signal table/columns (adjust here if your schema differs)
SIGNALS_TABLE = os.getenv("SIGNALS_TABLE", "signals")
COL_SYMBOL    = os.getenv("COL_SYMBOL", "symbol")
COL_SIDE      = os.getenv("COL_SIDE", "side")           # 'buy' or 'sell'
COL_STRENGTH  = os.getenv("COL_STRENGTH", "strength")   # float
COL_PORTFOLIO = os.getenv("COL_PORTFOLIO", "portfolio_id")
COL_CREATED   = os.getenv("COL_CREATED", "created_at")  # timestamptz

def log(msg: str, level: str = "INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} executor_bracket | {msg}", flush=True)

# ---------------- DB helpers ---------------- #

def _pg_conn():
    if DATABASE_URL:
        if HAVE_PSYCOPG3:
            return psycopg.connect(DATABASE_URL, autocommit=True)
        else:
            return psycopg2.connect(DATABASE_URL)  # autocommit default differs
    # Build from discrete vars
    if HAVE_PSYCOPG3:
        return psycopg.connect(
            host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE, port=PGPORT,
            autocommit=True
        )
    else:
        conn = psycopg2.connect(host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE, port=PGPORT)
        conn.autocommit = True
        return conn

def fetch_signals(since_days: int, min_strength: float, portfolio_id: int) -> List[Dict]:
    since_ts = datetime.now(timezone.utc) - timedelta(days=since_days)
    sql = f"""
        SELECT {COL_SYMBOL} AS symbol,
               {COL_SIDE}   AS side,
               {COL_STRENGTH} AS strength,
               {COL_CREATED}  AS created_at
        FROM {SIGNALS_TABLE}
        WHERE {COL_CREATED} >= %s
          AND {COL_STRENGTH} >= %s
          AND ({COL_PORTFOLIO} = %s OR %s IS NULL)
        ORDER BY {COL_CREATED} DESC
    """
    rows = []
    conn = _pg_conn()
    try:
        if HAVE_PSYCOPG3:
            with conn.cursor() as cur:
                cur.execute(sql, (since_ts, min_strength, portfolio_id, None))
                for r in cur.fetchall():
                    rows.append({
                        "symbol":  r[0].upper(),
                        "side":    r[1].lower(),
                        "strength": float(r[2]),
                        "created_at": r[3],
                    })
        else:
            with conn.cursor() as cur:
                cur.execute(sql, (since_ts, min_strength, portfolio_id, None))
                for r in cur.fetchall():
                    rows.append({
                        "symbol":  r[0].upper(),
                        "side":    r[1].lower(),
                        "strength": float(r[2]),
                        "created_at": r[3],
                    })
    finally:
        conn.close()
    return rows

# ---------------- Guardrails ---------------- #

def dedupe_recent(signals: List[Dict]) -> List[Dict]:
    """
    Keep the strongest recent signal per (symbol, side) within DEDUPE_MINS.
    """
    by_key: Dict[Tuple[str,str], Dict] = {}
    window = timedelta(minutes=DEDUPE_MINS)
    now = datetime.now(timezone.utc)
    for s in signals:
        key = (s["symbol"], s["side"])
        if (now - s["created_at"]).total_seconds() > window.total_seconds():
            continue
        prev = by_key.get(key)
        if (prev is None) or (s["strength"] > prev["strength"]):
            by_key[key] = s
    return list(by_key.values())

def wash_trade_lock(open_orders: List[Dict], symbol: str, side: str) -> bool:
    """
    Block if there's an opposite-side open order for the same symbol.
    """
    opp = "sell" if side == "buy" else "buy"
    for o in open_orders:
        if o.get("symbol") != symbol:
            continue
        if o.get("side") == opp:
            return True
    return False

def count_open_positions() -> int:
    r = requests.get(
        f"{ALPACA_BASE_URL}/v2/positions",
        headers={"APCA-API-KEY-ID": API_KEY or "", "APCA-API-SECRET-KEY": API_SECRET or ""},
        timeout=15,
    )
    if r.status_code >= 300:
        return 0
    try:
        return len(r.json())
    except Exception:
        return 0

# ---------------- Main flow ---------------- #

def place_from_signals(since_days: int, min_strength: float, portfolio_id: int, max_positions: int):
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    # 1) Pull signals
    sigs = fetch_signals(since_days, min_strength, portfolio_id)
    if not sigs:
        log("No signals found matching filters.")
        return

    # 2) Tight recent dedupe
    recent = dedupe_recent(sigs)
    if not recent:
        log("No signals within dedupe window.")
        return

    # 3) Respect max positions
    open_pos_count = count_open_positions()
    budget = max(0, max_positions - open_pos_count)
    if budget <= 0:
        log(f"Max positions reached ({open_pos_count}/{max_positions}); skipping.")
        return

    # 4) Pull open orders once for wash-trade check
    open_orders = list_open_orders()

    placed = 0
    for s in recent:
        if placed >= budget:
            break
        sym, side = s["symbol"], s["side"]
        if side not in ("buy","sell"):
            continue
        if wash_trade_lock(open_orders, sym, side):
            log(f"{sym} {side}: wash-trade lock active; skipping.", level="WARN")
            continue

        try:
            log(f"Submit bracket for {sym} {side} (strength={s['strength']:.2f})")
            submit_bracket_entry(sym, side, qty=None)  # qty=None -> tiny ATR-based auto-size
            placed += 1
        except Exception as e:
            log(f"{sym} {side}: submit failed -> {e}", level="ERROR")

    log(f"Done. Placed {placed} bracket order(s).")


def parse_cli(argv: List[str]):
    import argparse
    ap = argparse.ArgumentParser(description="Submit ATR-aware BRACKET orders from signals.")
    ap.add_argument("--since-days", type=int, default=SINCE_DAYS, help="Lookback window for signals (days).")
    ap.add_argument("--min-strength", type=float, default=MIN_STRENGTH, help="Minimum signal strength.")
    ap.add_argument("--portfolio-id", type=int, default=PORTFOLIO_ID, help="Portfolio id column filter.")
    ap.add_argument("--max-positions", type=int, default=MAX_POSITIONS, help="Cap total concurrent positions.")
    return ap.parse_args(argv)

if __name__ == "__main__":
    args = parse_cli(sys.argv[1:])
    place_from_signals(args.since_days, args.min_strength, args.portfolio_id, args.max_positions)

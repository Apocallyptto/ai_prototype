# services/executor_bracket.py
from __future__ import annotations

import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests

from services.bracket_helper import submit_bracket  # <- single, clean import

# ---- Env / config ----
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR executor_bracket | DATABASE_URL is not set.")
    sys.exit(1)

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")  # optional

# pull only very recent signals
EXECUTOR_SIGNAL_WINDOW_MIN = int(os.getenv("EXECUTOR_SIGNAL_WINDOW_MIN", "10"))

# risk/cap controls (enforced here in light form; you can keep your separate managers too)
WASH_LOCK_MIN = int(os.getenv("WASH_LOCK_MIN", "5"))
MAX_PARENTS_PER_SYMBOL = int(os.getenv("MAX_PARENTS_PER_SYMBOL", "2"))

# Broker endpoints (for sanity checks or future enhancements)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

S = requests.Session()
if API_KEY and API_SECRET:
    S.headers.update({
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Accept": "application/json",
        "Content-Type": "application/json",
    })

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _conn():
    return psycopg2.connect(DATABASE_URL)

# ---------- DB: fetch latest-per-symbol signals within window ----------

def fetch_latest_signals(symbols: List[str], min_strength: float, window_min: int, portfolio_id: Optional[int|str]) -> List[Dict]:
    """Return at most one row per symbol (latest), inside the time window and above min_strength."""
    since_dt = _utcnow() - timedelta(minutes=window_min)
    sql = """
        SELECT DISTINCT ON (symbol)
            symbol, side, strength, created_at, portfolio_id, source
        FROM public.signals
        WHERE symbol = ANY(%s)
          AND strength >= %s
          AND created_at >= %s
          {and_portfolio}
        ORDER BY symbol, created_at DESC
    """
    and_portfolio = ""
    params: List = [symbols, float(min_strength), since_dt]
    if portfolio_id:
        and_portfolio = "AND portfolio_id = %s"
        params.append(int(portfolio_id))

    sql = sql.format(and_portfolio=and_portfolio)

    with _conn() as c, c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    # keep ordering by created_at just for pleasant logs
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows

# ---------- Broker state helpers (light checks) ----------

def _list_open_parent_count(symbol: str) -> int:
    """Count open parent brackets for a symbol."""
    try:
        r = S.get(f"{ALPACA_BASE_URL}/v2/orders", params={
            "status": "open",
            "limit": 200,
            "nested": "true",
            "symbols": symbol,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Count orders marked as "bracket" parents (no parent_order_id and has take_profit/stop_loss legs)
        cnt = 0
        for o in data:
            if not o.get("parent_order_id") and o.get("order_class") == "bracket":
                cnt += 1
        return cnt
    except Exception:
        return 0

# (Optional) simplistic wash lock using recent submitted orders
def _wash_lock_active(symbol: str, side: str, minutes: int = WASH_LOCK_MIN) -> bool:
    try:
        r = S.get(f"{ALPACA_BASE_URL}/v2/orders", params={
            "status": "all",
            "limit": 50,
            "symbols": symbol,
        }, timeout=10)
        r.raise_for_status()
        data = r.json()
        cutoff = _utcnow() - timedelta(minutes=minutes)
        side = side.lower()
        for o in data:
            # treat recent same-side parent as lock
            if o.get("symbol") == symbol and (o.get("side") or "").lower() == side:
                ts = o.get("submitted_at") or o.get("created_at") or o.get("updated_at")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if dt >= cutoff:
                        return True
        return False
    except Exception:
        return False

# ---------- Placement ----------

def place_from_signals():
    rows = fetch_latest_signals(SYMBOLS, MIN_STRENGTH, EXECUTOR_SIGNAL_WINDOW_MIN, PORTFOLIO_ID)
    if not rows:
        print("INFO executor_bracket | No signals within window.")
        return

    placed = 0
    for r in rows:
        symbol = r["symbol"].upper()
        side = (r["side"] or "").lower().strip()
        strength = float(r["strength"])
        created = r["created_at"]
        source = r.get("source") or "unknown"

        if side not in ("buy", "sell"):
            print(f"WARN executor_bracket | {symbol} invalid side={side}; skipping.")
            continue

        # Light-cap checks
        if _wash_lock_active(symbol, side, WASH_LOCK_MIN):
            print(f"WARN executor_bracket | {symbol} {side}: wash-trade lock active; skipping.")
            continue

        open_cnt = _list_open_parent_count(symbol)
        if open_cnt >= MAX_PARENTS_PER_SYMBOL:
            print(f"INFO executor_bracket | {symbol} {side}: open-parent cap reached ({open_cnt}); skipping.")
            continue

        # Place the bracket (qty=None => dynamic sizing if enabled)
        try:
            resp = submit_bracket(
                symbol=symbol,
                side=side,
                strength=strength,
                qty=None,
                client_id=f"BRK-{symbol}-{int(time.time())}",
                time_in_force="day",
                order_type="market",
                extended_hours=False,  # day-only; brackets are RTH on Alpaca
            )
            oid = resp.get("id")
            print(f"INFO executor_bracket | Placed {symbol} {side} (strength={strength:.2f}) id={oid} source={source} at={created}")
            placed += 1
        except requests.HTTPError as he:
            print(f"ERROR executor_bracket | {symbol} {side}: submit failed -> {he} | body={getattr(he.response,'text', '')[:300]}")
        except Exception as e:
            print(f"ERROR executor_bracket | {symbol} {side}: unexpected error -> {e}")

    print(f"INFO executor_bracket | Done. Placed {placed} bracket order(s).")

# ---------- CLI ----------

def main():
    print(f"INFO executor_bracket | symbols={SYMBOLS} min_strength={MIN_STRENGTH} window_min={EXECUTOR_SIGNAL_WINDOW_MIN} portfolio_id={PORTFOLIO_ID or 'ANY'}")
    place_from_signals()

if __name__ == "__main__":
    main()

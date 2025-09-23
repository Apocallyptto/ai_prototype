# services/executor.py
"""
Minimal trade executor for Alpaca (paper by default).

Reads the latest signal per symbol from the DB and:
- if 'buy': open/keep a long position (market order)
- if 'sell': close any existing long position (market order)

Writes orders back to the `orders` table.

ENV VARS (required):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD   # Postgres
  ALPACA_API_KEY, ALPACA_API_SECRET
  ALPACA_BASE_URL      (e.g. https://paper-api.alpaca.markets)
  TRADING_MODE         'paper' or 'live'  (if unset => DRY RUN)

Optional:
  DEFAULT_QTY          default shares per trade (int, default 1)
"""

from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime, timezone

import requests
import pandas as pd
import sqlalchemy as sa


# ---------- Config helpers ----------

def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return v


def db_engine() -> sa.Engine:
    host = _require("DB_HOST")
    port = int(os.getenv("DB_PORT", "5432"))
    name = _require("DB_NAME")
    user = _require("DB_USER")
    pw   = _require("DB_PASSWORD")
    ssl  = os.getenv("DB_SSLMODE", "require")

    url = sa.engine.URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=pw,
        host=host,
        port=port,
        database=name,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)


def alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID": _require("ALPACA_API_KEY"),
        "APCA-API-SECRET-KEY": _require("ALPACA_API_SECRET"),
        "Content-Type": "application/json",
    }


def alpaca_base() -> str:
    return os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


def trading_enabled() -> bool:
    return os.getenv("TRADING_MODE", "").lower() in {"paper", "live"}


DEFAULT_QTY = int(os.getenv("DEFAULT_QTY", "1"))


# ---------- DB I/O ----------

def latest_signals(conn, since_days: int = 5) -> pd.DataFrame:
    """
    Return the latest signal per symbol from the last N days.
    Expected columns in `signals`: ts (timestamptz), symbol, signal ('buy'/'sell'), strength (float)
    """
    sql = sa.text("""
        with ranked as (
          select
            ts, coalesce(symbol, ticker) as symbol,
            coalesce(signal, side) as signal,
            strength,
            row_number() over (partition by coalesce(symbol, ticker) order by ts desc) as rn
          from signals
          where ts >= now() - (:days || ' days')::interval
        )
        select ts, symbol, signal, strength
        from ranked
        where rn = 1
        order by symbol
    """)
    df = pd.read_sql(sql, conn, params={"days": int(since_days)})
    # normalize
    df["symbol"] = df["symbol"].str.upper()
    df["signal"] = df["signal"].str.lower()
    return df


def insert_order_row(conn, *, symbol: str, side: str, qty: int, order_type: str,
                     limit_price: float | None, status: str,
                     filled_at: datetime | None, broker_order_id: str | None):
    """
    Inserts a row into orders table. Table is expected to have at least:
    ts, symbol, side, qty, order_type, limit_price, status, filled_at, broker_order_id
    (extra columns are fine)
    """
    ts = datetime.now(timezone.utc)
    conn.execute(sa.text("""
        insert into orders (ts, symbol, side, qty, order_type, limit_price, status, filled_at, broker_order_id)
        values (:ts, :symbol, :side, :qty, :order_type, :limit_price, :status, :filled_at, :broker_order_id)
    """), {
        "ts": ts, "symbol": symbol, "side": side, "qty": int(qty),
        "order_type": order_type, "limit_price": limit_price,
        "status": status, "filled_at": filled_at,
        "broker_order_id": broker_order_id
    })


# ---------- Alpaca calls ----------

def get_position(symbol: str) -> dict | None:
    r = requests.get(f"{alpaca_base()}/v2/positions/{symbol}",
                     headers=alpaca_headers(), timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def place_market_order(symbol: str, side: str, qty: int) -> dict:
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,               # 'buy' or 'sell'
        "type": "market",
        "time_in_force": "day"
    }
    r = requests.post(f"{alpaca_base()}/v2/orders",
                      headers=alpaca_headers(), data=json.dumps(payload), timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_order(order_id: str) -> dict:
    r = requests.get(f"{alpaca_base()}/v2/orders/{order_id}",
                     headers=alpaca_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


# ---------- Strategy logic ----------

def act_on_signal(conn, *, symbol: str, signal: str, qty: int = DEFAULT_QTY, dry_run: bool = False):
    """
    Simple policy:
      - if 'buy': ensure we hold a long position (submit market buy qty if flat)
      - if 'sell': close any long position (submit market sell all shares)
    """
    signal = signal.lower()
    symbol = symbol.upper()
    print(f"[signal] {symbol}: {signal}, qty={qty}")

    # Current position
    pos = None
    try:
        pos = get_position(symbol)
    except requests.HTTPError as e:
        # 404 means no position; other errors bubble up
        if e.response is None or e.response.status_code != 404:
            raise

    long_qty = int(float(pos["qty"])) if pos and float(pos.get("qty", 0)) > 0 else 0
    print(f"[position] {symbol}: long_qty={long_qty}")

    if signal == "buy":
        if long_qty > 0:
            print(f"[skip] Already long {symbol} ({long_qty}), no action.")
            return
        if dry_run:
            print(f"[dry-run] market buy {qty} {symbol}")
            insert_order_row(conn, symbol=symbol, side="buy", qty=qty,
                             order_type="market", limit_price=None,
                             status="simulated", filled_at=None, broker_order_id=None)
            return
        # Place order
        resp = place_market_order(symbol, "buy", qty)
        broker_id = resp.get("id")
        status = resp.get("status", "submitted")
        print(f"[order] BUY submitted id={broker_id} status={status}")
        # Small poll to get fill time if it fills quickly
        filled_at = None
        try:
            time.sleep(1.0)
            o = fetch_order(broker_id)
            if o.get("filled_at"):
                filled_at = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
                status = o.get("status", status)
        except Exception:
            pass
        insert_order_row(conn, symbol=symbol, side="buy", qty=qty,
                         order_type="market", limit_price=None,
                         status=status, filled_at=filled_at, broker_order_id=broker_id)

    elif signal == "sell":
        if long_qty <= 0:
            print(f"[skip] Not long {symbol}, no action.")
            return
        if dry_run:
            print(f"[dry-run] market sell {long_qty} {symbol}")
            insert_order_row(conn, symbol=symbol, side="sell", qty=long_qty,
                             order_type="market", limit_price=None,
                             status="simulated", filled_at=None, broker_order_id=None)
            return
        resp = place_market_order(symbol, "sell", long_qty)
        broker_id = resp.get("id")
        status = resp.get("status", "submitted")
        print(f"[order] SELL submitted id={broker_id} status={status}")
        filled_at = None
        try:
            time.sleep(1.0)
            o = fetch_order(broker_id)
            if o.get("filled_at"):
                filled_at = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00"))
                status = o.get("status", status)
        except Exception:
            pass
        insert_order_row(conn, symbol=symbol, side="sell", qty=long_qty,
                         order_type="market", limit_price=None,
                         status=status, filled_at=filled_at, broker_order_id=broker_id)

    else:
        print(f"[warn] Unknown signal '{signal}' for {symbol}; skipping.")


def main():
    import argparse
    p = argparse.ArgumentParser(description="Execute trades at Alpaca based on latest signals.")
    p.add_argument("--symbol", help="Only act on this symbol (e.g., AAPL). Default: all symbols with recent signals.")
    p.add_argument("--qty", type=int, default=DEFAULT_QTY, help=f"Order qty (default {DEFAULT_QTY})")
    p.add_argument("--since-days", type=int, default=5, help="Look back window for latest signals.")
    args = p.parse_args()

    dry_run = not trading_enabled()
    mode = os.getenv("TRADING_MODE", "").lower() or "DRY"
    print(f"Executor mode: {mode}  dry_run={dry_run}")

    eng = db_engine()
    with eng.begin() as conn:
        df = latest_signals(conn, since_days=args.since_days)
        if args.symbol:
            df = df[df["symbol"].str.upper() == args.symbol.upper()]
        if df.empty:
            print("No recent signals found.")
            return

        for _, row in df.iterrows():
            act_on_signal(conn,
                          symbol=row["symbol"],
                          signal=row["signal"],
                          qty=args.qty,
                          dry_run=dry_run)

    print("Done.")


if __name__ == "__main__":
    main()

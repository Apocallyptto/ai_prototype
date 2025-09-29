# services/executor.py
from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from typing import Iterable, List, Tuple

import sqlalchemy as sa
import pandas as pd

from lib.db import make_engine
from lib.broker_alpaca import place_marketable_limit, latest_price

# ---------- Logging ----------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("executor")

# ---------- Config (env) ----------------------------------------------------
PORTFOLIO_ID   = int(os.getenv("PORTFOLIO_ID", "1"))
MIN_STRENGTH   = float(os.getenv("MIN_STRENGTH", "0.30"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.05"))   # fraction of equity (fallback sizing)
MAX_POSITIONS  = int(os.getenv("MAX_POSITIONS", "10"))
MAX_QTY        = int(os.getenv("MAX_QTY_PER_ORDER", "5"))
TICKER_FILTER  = [s.strip().upper() for s in os.getenv("SIGNAL_TICKERS", "").split(",") if s.strip()]
DRY_RUN        = os.getenv("DRY_RUN", "").lower() in {"1","true","yes"}

# --------- Helpers -----------------------------------------------------------
def load_recent_signals(conn, since_days:int) -> pd.DataFrame:
    """
    Load signals newer than since_days and above MIN_STRENGTH.
    Optionally filter tickers if SIGNAL_TICKERS is set.
    """
    base_sql = """
        SELECT s.ts, s.ticker, s.timeframe, s.model, s.side, s.strength
        FROM signals s
        WHERE s.ts >= now() - interval :since
          AND s.strength >= :min_strength
        ORDER BY s.ts DESC
    """
    df = pd.read_sql(
        sa.text(base_sql),
        conn,
        params={"since": f"{since_days} days", "min_strength": MIN_STRENGTH},
    )
    if TICKER_FILTER:
        df = df[df["ticker"].isin(TICKER_FILTER)]
    return df.reset_index(drop=True)

def _portfolio_equity(conn) -> float:
    """
    Best-effort equity: last daily_pnl.equity for the portfolio.
    Fallback to 10_000 if not found.
    """
    q = """
        SELECT equity
        FROM daily_pnl
        WHERE portfolio_id = :pid
        ORDER BY "date" DESC
        LIMIT 1
    """
    try:
        row = conn.execute(sa.text(q), {"pid": PORTFOLIO_ID}).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        logger.debug("equity fetch fail: %s", e)
    return 10_000.0

def _current_open_positions(conn) -> int:
    q = "SELECT COUNT(*) FROM orders WHERE status IN ('submitted','pending_new','accepted','new')"
    try:
        n = conn.execute(sa.text(q)).scalar()
        return int(n or 0)
    except Exception:
        return 0

def _sized_qty(ticker: str, equity: float) -> int:
    """
    Position size by RISK_PER_TRADE fraction of equity / last price,
    clamped to [1, MAX_QTY].
    """
    try:
        px = latest_price(ticker)
        qty = max(1, int((equity * RISK_PER_TRADE) // px))
    except Exception as e:
        logger.debug("price sizing failed for %s: %s; defaulting to 1", ticker, e)
        qty = 1

    return max(1, min(qty, MAX_QTY))

def create_order_row(conn, ticker: str, side: str, qty: int, lim: float, status_text: str):
    conn.execute(sa.text("""
        INSERT INTO orders
            (ts, symbol_id, ticker, side, qty, order_type, limit_price, status)
        VALUES
            (now(),
             (SELECT id FROM symbols WHERE ticker = :ticker),
             :ticker, :side, :qty, 'limit', :lim, :status)
    """), {
        "ticker": ticker,
        "side": side,
        "qty": qty,
        "lim": lim,
        "status": status_text
    })

# --------- Main execution ----------------------------------------------------
def run(since_days:int, dry_run:bool=False):
    eng = make_engine()
    with eng.begin() as conn:
        # 1) Load candidate signals
        sigs = load_recent_signals(conn, since_days)
        if sigs.empty:
            logger.info("No recent signals (>= %.2f).", MIN_STRENGTH)
            return

        logger.info("Loaded %d signals (min_strength=%.2f).", len(sigs), MIN_STRENGTH)

        # 2) Capacity check
        open_n = _current_open_positions(conn)
        if open_n >= MAX_POSITIONS:
            logger.info("Max open positions reached (%d). No new orders.", MAX_POSITIONS)
            return

        # 3) One pass, place orders for top signals per ticker/timeframe
        equity = _portfolio_equity(conn)

        # You can add smarter dedupe here; keep it simple: latest rows first
        placed = 0
        for _, row in sigs.iterrows():
            if placed + open_n >= MAX_POSITIONS:
                logger.info("Reached position cap while iterating.")
                break

            ticker = row["ticker"]
            side   = row["side"]
            strength = float(row["strength"])

            qty = _sized_qty(ticker, equity)
            logger.info("Signal %s %s str=%.3f qty=%s", ticker, side, strength, qty)

            if dry_run or DRY_RUN:
                # simulate broker, still show in logs (do NOT write to DB on dry-run)
                try:
                    from lib.broker_alpaca import latest_price
                    last = latest_price(ticker)
                    lim  = round(last * (1.05 if side=="buy" else 0.95), 2)
                except Exception:
                    lim = None
                logger.info("[DRY-RUN] would place limit DAY extended %s %s x%s @%s",
                            side, ticker, qty, lim)
                placed += 1
                continue

            # 4) Send to broker (marketable DAY+LIMIT)
            result = place_marketable_limit(
                ticker, side, qty,
                # these may also come from env via lib.broker_alpaca defaults
                pad_up=1.05, pad_down=0.95, extended_hours=True,
            )

            status_text = "submitted"
            if result["http_status"] == 200 and result["json"]:
                status_text = result["json"].get("status", "submitted")

            # 5) Persist to orders table for the UI
            try:
                create_order_row(conn, ticker, side, qty, float(result["limit_price"]), status_text)
                placed += 1
            except Exception as e:
                logger.error("DB insert failed for %s %s: %s", ticker, side, e)

            # 6) Log details
            logger.info("alpaca order -> %s | last=%.2f lim=%.2f | %s",
                        result["http_status"], result["last"], result["limit_price"], result["text"])

        logger.info("Done. Placed %d order(s).", placed)

# --------- CLI ---------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute signals -> Alpaca orders")
    p.add_argument("--since-days", type=int, default=3, help="lookback window for signals")
    p.add_argument("--dry-run", action="store_true", help="simulate broker orders only")
    return p.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.since_days, dry_run=args.dry_run)

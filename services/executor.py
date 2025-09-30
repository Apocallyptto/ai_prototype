# services/executor.py
import os
import argparse
import logging
import datetime as dt
from datetime import timezone

import pandas as pd
import sqlalchemy as sa

from lib.db import make_engine
from lib.broker_alpaca import place_marketable_limit

# -------- logging ----------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("executor")


# -------- config via env ----------
MIN_STRENGTH = float(os.environ.get("MIN_STRENGTH", "0.30"))
SIGNAL_TICKERS = [
    t.strip().upper() for t in os.environ.get("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",") if t.strip()
]
MAX_POSITIONS = int(os.environ.get("MAX_POSITIONS", "10"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.05"))

# simple constant size while testing
DEFAULT_QTY = int(os.environ.get("TEST_QTY", "1"))

# aggressiveness for marketable limit
PAD_UP = float(os.environ.get("PAD_UP", "1.05"))
PAD_DOWN = float(os.environ.get("PAD_DOWN", "0.95"))


def fetch_signals(conn, since_days: int) -> pd.DataFrame:
    q = sa.text(
        """
        SELECT ts, ticker, timeframe, model, side, strength
        FROM signals
        WHERE ts >= now() - (:d || ' days')::interval
        ORDER BY ts ASC
        """
    )
    df = pd.read_sql(q, conn, params={"d": since_days})
    if SIGNAL_TICKERS:
        df = df[df["ticker"].isin(SIGNAL_TICKERS)]
    df = df[df["strength"] >= MIN_STRENGTH]
    return df


def have_recent_pending(conn, ticker: str, side: str) -> bool:
    row = conn.execute(
        sa.text(
            """
            SELECT 1
            FROM orders
            WHERE ticker=:t AND side=:s
              AND status IN ('pending_new','submitted')
              AND ts >= now() - interval '10 minutes'
            LIMIT 1
            """
        ),
        {"t": ticker, "s": side},
    ).fetchone()
    return bool(row)


def insert_order_row(conn, ticker: str, side: str, qty: int, lim: float, status_text: str, ts_iso: str):
    conn.execute(
        sa.text(
            """
            INSERT INTO orders
                (ts, symbol_id, ticker, side, qty, order_type, limit_price, status)
            VALUES
                (:ts,
                 (SELECT id FROM symbols WHERE ticker = :ticker),
                 :ticker, :side, :qty, 'limit', :lim, :status)
            """
        ),
        {
            "ts": ts_iso,
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "lim": lim,
            "status": status_text,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=int, default=3, help="how many days back to read signals")
    args = ap.parse_args()

    eng = make_engine()
    placed = 0

    with eng.begin() as conn:
        sigs = fetch_signals(conn, args.since_days)

        # crude position cap
        open_positions = pd.read_sql("SELECT COUNT(*) AS n FROM positions", conn).iloc[0]["n"] if "positions" in eng.dialect.get_table_names(conn) else 0  # type: ignore

        for _, row in sigs.iterrows():
            ticker = str(row["ticker"]).upper()
            side = str(row["side"]).lower()
            qty = DEFAULT_QTY

            if have_recent_pending(conn, ticker, side):
                logger.info("skip %s %s: recent pending order exists", ticker, side)
                continue

            if open_positions >= MAX_POSITIONS and side == "buy":
                logger.info("skip %s buy: max positions reached", ticker)
                continue

            # place marketable DAY+LIMIT
            res = place_marketable_limit(
                ticker,
                side,
                qty,
                pad_up=PAD_UP,
                pad_down=PAD_DOWN,
                extended_hours=True,
            )

            status_text = "submitted"
            if res["http_status"] == 200 and res["json"]:
                status_text = res["json"].get("status", "submitted")

            # timezone-aware UTC
            ts_iso = dt.datetime.now(timezone.utc).isoformat()

            try:
                insert_order_row(conn, ticker, side, qty, res["limit_price"], status_text, ts_iso)
            except Exception as e:
                logger.exception("DB insert failed for %s %s", ticker, side)

            placed += 1

    logger.info("Done. Placed %d order(s).", placed)


if __name__ == "__main__":
    main()

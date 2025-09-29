# services/executor.py
from __future__ import annotations

import os
import argparse
import datetime as dt
import logging
from typing import Optional

import pandas as pd
import sqlalchemy as sa

from lib.db import make_engine
from lib.db_orders import log_order_row
from lib.broker_alpaca import place_marketable_limit


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def get_signals(since_days: int, tickers_csv: Optional[str]) -> pd.DataFrame:
    eng = make_engine()
    where_tickers = ""
    params = {"days": since_days}
    if tickers_csv:
        tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
        where_tickers = "AND s.ticker = ANY(:tickers)"
        params["tickers"] = tickers

    q = sa.text(
        f"""
        SELECT s.ts, s.ticker, s.timeframe, s.model, s.side, s.strength
        FROM signals s
        WHERE s.ts >= now() - (:days || ' days')::interval
          {where_tickers}
        ORDER BY s.ts ASC
        """
    )
    with make_engine().connect() as c:
        df = pd.read_sql(q, c, params=params)
    return df


def should_trade(row: pd.Series, min_strength: float) -> bool:
    try:
        return float(row["strength"]) >= min_strength
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-days", type=int, default=3, help="look-back window for signals")
    parser.add_argument("--dry-run", action="store_true", help="no broker orders, only log")
    args = parser.parse_args()

    # --- logging
    level = env_str("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("executor")

    min_strength = env_float("MIN_STRENGTH", 0.30)
    signal_tickers = os.environ.get("SIGNAL_TICKERS", "")  # optional filter like "AAPL,MSFT,SPY"
    pad_up = env_float("PAD_UP", 1.05)
    pad_down = env_float("PAD_DOWN", 0.95)

    df = get_signals(args.since_days, signal_tickers)
    if df.empty:
        logger.info("No signals in the last %s day(s).", args.since_days)
        return

    placed = 0
    last_ts_by_ticker = {}  # tiny cooldown: one trade per ticker per run
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper()
        side = str(row["side"]).lower().strip()  # "buy" or "sell"
        strength = float(row["strength"])
        qty = 1  # simple demo sizing; wire in your sizing if you prefer

        if not should_trade(row, min_strength):
            continue

        # one per ticker per run (prevent rapid flip-flopping from back-to-back signals)
        if ticker in last_ts_by_ticker:
            continue
        last_ts_by_ticker[ticker] = row["ts"]

        if args.dry_run:
            logger.info("[DRY] %s %s qty=%s str=%.3f", side, ticker, qty, strength)
            continue

        # --- broker call (DAY+LIMIT, “marketable” price)
        res = place_marketable_limit(
            ticker,
            side,
            qty,
            pad_up=pad_up,
            pad_down=pad_down,
            extended_hours=True,
        )

        http_status = res.get("http_status")
        last_px = res.get("last")
        lim_px = res.get("limit_price")
        txt = res.get("text", "")

        status_text = "error"
        ts_iso = dt.datetime.utcnow().isoformat() + "Z"
        if http_status == 200 and res.get("json"):
            j = res["json"]
            status_text = j.get("status", "submitted")
            ts_iso = j.get("submitted_at") or j.get("created_at") or ts_iso

        ok = log_order_row(
            ts_iso=ts_iso,
            ticker=ticker,
            side=side,
            qty=float(qty),
            order_type="limit",
            limit_price=float(lim_px) if lim_px is not None else None,
            status=status_text,
            logger=logger,
        )

        if not ok:
            logger.error("Order logged with error for %s %s", ticker, side)

        logger.info(
            "alpaca order -> %s | last=%s lim=%s | %s",
            http_status,
            f"{last_px:.2f}" if isinstance(last_px, (int, float)) else last_px,
            f"{lim_px:.2f}" if isinstance(lim_px, (int, float)) else lim_px,
            txt[:300],
        )
        placed += 1

    logger.info("Done. Placed %s order(s).", placed)


if __name__ == "__main__":
    main()

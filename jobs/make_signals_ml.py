#!/usr/bin/env python3
"""
jobs/make_signals_ml.py

Stable signal maker without `ml.*` dependency.

- Pulls recent 5-min bars from Alpaca Market Data
- Writes signals into Postgres

Signals:
- BUY if last_close > SMA
- SELL if last_close < SMA  (executor may skip if LONG_ONLY=True)

Strength:
- abs(close - sma) / sma  (clamped 0..1)

Env:
- SYMBOLS="AAPL,MSFT,SPY"
- PORTFOLIO_ID=1
- MIN_STRENGTH=0.60
- SIGNAL_POLL_SECONDS=60
- BARS_LIMIT=120
- SMA_PERIOD=50
- DB_URL or DATABASE_URL
- ALPACA_API_KEY / ALPACA_API_SECRET
"""

import os
import time
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


LOG = logging.getLogger("signal_maker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _normalize_sqlalchemy_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        raise RuntimeError("DB_URL / DATABASE_URL not set")

    # if plain postgresql://, force psycopg (v3) driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_engine():
    raw = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or ""
    url = _normalize_sqlalchemy_url(raw)
    return create_engine(url, pool_pre_ping=True, future=True)


def _split_symbols() -> list[str]:
    s = os.getenv("SYMBOLS", "AAPL,MSFT,SPY")
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def _sma(values: list[float], period: int) -> float | None:
    if period <= 1 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / float(period)


def _insert_signal(engine, symbol: str, side: str, strength: float, price: float, portfolio_id: int) -> None:
    with engine.begin() as con:
        con.execute(
            text(
                """
                INSERT INTO signals (symbol, side, strength, price, portfolio_id)
                VALUES (:symbol, :side, :strength, :price, :portfolio_id)
                """
            ),
            {
                "symbol": symbol,
                "side": side,
                "strength": float(strength),
                "price": float(price),
                "portfolio_id": int(portfolio_id),
            },
        )


def main() -> None:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    poll = int(os.getenv("SIGNAL_POLL_SECONDS", "60"))
    min_strength = float(os.getenv("MIN_STRENGTH", "0.60"))
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))
    bars_limit = int(os.getenv("BARS_LIMIT", "120"))
    sma_period = int(os.getenv("SMA_PERIOD", "50"))

    symbols = _split_symbols()
    engine = _get_engine()

    data_client = StockHistoricalDataClient(api_key, api_secret)

    tf = TimeFrame(5, TimeFrameUnit.Minute)  # ✅ FIX: 5-min timeframe

    LOG.info(
        "signal_maker starting | symbols=%s | poll=%ss | min_strength=%.4f | bars_limit=%s | sma_period=%s | portfolio_id=%s",
        symbols, poll, min_strength, bars_limit, sma_period, portfolio_id
    )

    while True:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                limit=bars_limit,
            )

            bars = data_client.get_stock_bars(req)
            df = getattr(bars, "df", None)

            if df is None or df.empty:
                LOG.info("no_bars | sleep=%ss", poll)
                time.sleep(poll)
                continue

            inserted = 0
            for sym in symbols:
                try:
                    sym_df = df.loc[sym]  # MultiIndex: (symbol, timestamp)
                except Exception:
                    continue

                if sym_df is None or sym_df.empty:
                    continue

                closes = [float(x) for x in sym_df["close"].tolist() if x is not None]
                if len(closes) < max(10, sma_period):
                    continue

                close = float(closes[-1])
                sma = _sma(closes, sma_period)
                if sma is None or sma <= 0:
                    continue

                side = "buy" if close > sma else "sell"
                strength = _clamp01(abs(close - sma) / sma)

                if strength < min_strength:
                    continue

                _insert_signal(engine, sym, side, strength, close, portfolio_id)
                inserted += 1

            LOG.info("cycle_done | inserted=%s | sleep=%ss", inserted, poll)

        except Exception as e:
            LOG.exception("signal_maker_error: %r", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()

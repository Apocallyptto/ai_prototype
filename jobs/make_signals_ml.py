#!/usr/bin/env python3
"""
jobs/make_signals_ml.py

Stable signal maker without `ml.*` dependency.

- Pulls recent 5-min bars from Alpaca Market Data
- Writes signals into Postgres

Signal:
- BUY if last_close > SMA
- SELL if last_close < SMA

Strength scaling:
- pct_diff = abs(close - sma) / sma
- strength = clamp01(pct_diff / STRENGTH_PCT_FOR_1)

Anti-spam / stability:
- SIGNAL_DEDUPE_SECONDS: do not insert same (symbol, side, portfolio_id) too often
- INSERT_TOP_IF_NONE: if no candidate passes MIN_STRENGTH, insert best candidate anyway (testing)
- SIGNAL_SIDE_OVERRIDE=buy|sell: force side (testing)

Env:
- SYMBOLS="AAPL,MSFT"
- PORTFOLIO_ID=1
- MIN_STRENGTH=0.35
- SIGNAL_POLL_SECONDS=60
- BARS_LIMIT=120
- SMA_PERIOD=50
- STRENGTH_PCT_FOR_1=0.002
- SIGNAL_DEDUPE_SECONDS=300
- INSERT_TOP_IF_NONE=0|1
- SIGNAL_SIDE_OVERRIDE=buy|sell|""
- DEBUG_SIGNALS=0|1
- DB_URL or DATABASE_URL
- ALPACA_API_KEY / ALPACA_API_SECRET
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine, text

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


LOG = logging.getLogger("signal_maker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = (v or "").strip().lower()
    return v not in ("0", "false", "no", "off")


def _normalize_sqlalchemy_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        raise RuntimeError("DB_URL / DATABASE_URL not set")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_engine():
    raw = os.getenv("DB_URL") or os.getenv("DATABASE_URL") or ""
    url = _normalize_sqlalchemy_url(raw)
    return create_engine(url, pool_pre_ping=True, future=True)


def _split_symbols() -> list[str]:
    s = os.getenv("SYMBOLS", "AAPL,MSFT")
    return [x.strip().upper() for x in s.split(",") if x.strip()]


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return x


def _sma(values: list[float], period: int) -> Optional[float]:
    if period <= 1 or len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / float(period)


def _recent_signal_exists(engine, symbol: str, side: str, portfolio_id: int, dedupe_seconds: int) -> bool:
    if dedupe_seconds <= 0:
        return False
    with engine.begin() as con:
        r = con.execute(
            text(
                """
                SELECT 1
                FROM signals
                WHERE symbol=:symbol
                  AND side=:side
                  AND portfolio_id=:pid
                  AND created_at > (NOW() - (:sec || ' seconds')::interval)
                LIMIT 1
                """
            ),
            {"symbol": symbol, "side": side, "pid": int(portfolio_id), "sec": int(dedupe_seconds)},
        ).fetchone()
        return bool(r)


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


@dataclass
class Candidate:
    symbol: str
    side: str
    strength: float
    price: float
    sma: float
    pct_diff: float


def main() -> None:
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    poll = int(os.getenv("SIGNAL_POLL_SECONDS", "60"))
    min_strength = float(os.getenv("MIN_STRENGTH", "0.35"))
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))
    bars_limit = int(os.getenv("BARS_LIMIT", "120"))
    sma_period = int(os.getenv("SMA_PERIOD", "50"))
    strength_pct_for_1 = float(os.getenv("STRENGTH_PCT_FOR_1", "0.002"))
    dedupe_seconds = int(os.getenv("SIGNAL_DEDUPE_SECONDS", "300"))
    insert_top_if_none = _env_bool("INSERT_TOP_IF_NONE", False)
    debug = _env_bool("DEBUG_SIGNALS", False)

    side_override = (os.getenv("SIGNAL_SIDE_OVERRIDE") or "").strip().lower()
    if side_override not in ("", "buy", "sell"):
        raise RuntimeError("SIGNAL_SIDE_OVERRIDE must be '', 'buy', or 'sell'")

    if strength_pct_for_1 <= 0:
        raise RuntimeError("STRENGTH_PCT_FOR_1 must be > 0")

    symbols = _split_symbols()
    engine = _get_engine()
    data_client = StockHistoricalDataClient(api_key, api_secret)
    tf = TimeFrame(5, TimeFrameUnit.Minute)

    LOG.info(
        "signal_maker starting | symbols=%s | poll=%ss | min_strength=%.4f | bars_limit=%s | sma_period=%s | "
        "portfolio_id=%s | strength_pct_for_1=%s | dedupe_seconds=%s | insert_top_if_none=%s | side_override=%s | debug=%s",
        symbols, poll, min_strength, bars_limit, sma_period, portfolio_id,
        strength_pct_for_1, dedupe_seconds, insert_top_if_none, side_override or "none", debug
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

            # For debug: what did Alpaca actually return?
            try:
                available_syms = sorted(set(df.index.get_level_values(0)))
            except Exception:
                available_syms = []

            inserted = 0
            best: Optional[Candidate] = None

            for sym in symbols:
                try:
                    sym_df = df.loc[sym]
                except Exception:
                    if debug:
                        LOG.info("debug | %s | no_df_for_symbol | available=%s", sym, available_syms)
                    continue

                if sym_df is None or sym_df.empty:
                    if debug:
                        LOG.info("debug | %s | empty_df", sym)
                    continue

                closes = [float(x) for x in sym_df["close"].tolist() if x is not None]
                if len(closes) < max(10, sma_period):
                    if debug:
                        LOG.info("debug | %s | not_enough_closes=%s need=%s", sym, len(closes), sma_period)
                    continue

                close = float(closes[-1])
                sma = _sma(closes, sma_period)
                if sma is None or sma <= 0:
                    if debug:
                        LOG.info("debug | %s | bad_sma=%s", sym, sma)
                    continue

                side = side_override if side_override else ("buy" if close > sma else "sell")
                pct_diff = abs(close - sma) / sma
                strength = _clamp01(pct_diff / strength_pct_for_1)

                if debug:
                    LOG.info(
                        "debug | %s | close=%.4f sma=%.4f pct_diff=%.6f strength=%.4f side=%s",
                        sym, close, sma, pct_diff, strength, side
                    )

                cand = Candidate(sym, side, strength, close, sma, pct_diff)
                if best is None or cand.strength > best.strength:
                    best = cand

                # main threshold path
                if strength < min_strength:
                    continue

                if _recent_signal_exists(engine, sym, side, portfolio_id, dedupe_seconds):
                    if debug:
                        LOG.info("debug | %s | %s | skipped_by_dedupe=%ss", sym, side, dedupe_seconds)
                    continue

                _insert_signal(engine, sym, side, strength, close, portfolio_id)
                inserted += 1

            # fallback for testing
            if inserted == 0 and insert_top_if_none and best is not None:
                if not _recent_signal_exists(engine, best.symbol, best.side, portfolio_id, dedupe_seconds):
                    _insert_signal(engine, best.symbol, best.side, best.strength, best.price, portfolio_id)
                    inserted += 1
                    LOG.info(
                        "fallback_insert | symbol=%s side=%s strength=%.4f pct_diff=%.6f (no candidate met min_strength=%.4f)",
                        best.symbol, best.side, best.strength, best.pct_diff, min_strength
                    )

            LOG.info("cycle_done | inserted=%s | sleep=%ss", inserted, poll)

        except Exception as e:
            LOG.exception("signal_maker_error: %r", e)

        time.sleep(poll)


if __name__ == "__main__":
    main()

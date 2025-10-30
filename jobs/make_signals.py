# jobs/make_signals.py
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa

from lib.db import make_engine  # already in your repo

# optional providers — imported lazily so the container can run without them
_ALPACA_READY = None
_YF_READY = None

log = logging.getLogger("make_signals")
logging.basicConfig(level=logging.INFO)

# ------------ schema helpers ------------

_SIGNALS_HAS_JSONB: Optional[bool] = None
_SIGNALS_HAS_PX: Optional[bool] = None

def signals_has_jsonb(conn) -> bool:
    global _SIGNALS_HAS_JSONB
    if _SIGNALS_HAS_JSONB is not None:
        return _SIGNALS_HAS_JSONB
    q = sa.text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='signals'
          AND column_name='signal'
          AND data_type='jsonb'
        LIMIT 1
    """)
    _SIGNALS_HAS_JSONB = bool(conn.execute(q).scalar())
    return _SIGNALS_HAS_JSONB

def signals_has_px(conn) -> bool:
    global _SIGNALS_HAS_PX
    if _SIGNALS_HAS_PX is not None:
        return _SIGNALS_HAS_PX
    q = sa.text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name='signals'
          AND column_name='px'
        LIMIT 1
    """)
    _SIGNALS_HAS_PX = bool(conn.execute(q).scalar())
    return _SIGNALS_HAS_PX

def get_or_create_symbol(conn, ticker: str, name: str | None = None) -> int:
    q = sa.text("""
        INSERT INTO symbols (ticker, name)
        VALUES (:ticker, COALESCE(:name, :ticker))
        ON CONFLICT (ticker) DO UPDATE
            SET name = COALESCE(EXCLUDED.name, symbols.name)
        RETURNING id
    """)
    return conn.execute(q, {"ticker": ticker, "name": name}).scalar_one()

# ------------ price fetchers ------------

def _alpaca_price(ticker: str) -> Optional[float]:
    """Try Alpaca IEX/SIP latest quote if keys are present."""
    global _ALPACA_READY
    if _ALPACA_READY is False:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        key = os.getenv("ALPACA_API_KEY")
        sec = os.getenv("ALPACA_API_SECRET")
        if not key or not sec:
            _ALPACA_READY = False
            return None
        cli = StockHistoricalDataClient(key, sec)
        req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        q = cli.get_stock_latest_quote(req)
        quote = q[ticker]
        px = quote.last_price or quote.ask_price or quote.bid_price
        if px is None:
            return None
        _ALPACA_READY = True
        return float(px)
    except Exception as e:
        log.debug("Alpaca price failed for %s: %s", ticker, e)
        _ALPACA_READY = False
        return None

def _yf_price(ticker: str) -> Optional[float]:
    """Try Yahoo fallback (best-effort)."""
    global _YF_READY
    if _YF_READY is False:
        return None
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info  # fast path when available
        px = getattr(info, "last_price", None) if hasattr(info, "last_price") else None
        if px is None:
            # slow path: recent 1d/1m bar
            bars = t.history(period="1d", interval="1m")
            if not bars.empty:
                px = float(bars["Close"].iloc[-1])
        if px is None:
            _YF_READY = False
            return None
        _YF_READY = True
        return float(px)
    except Exception as e:
        log.debug("Yahoo price failed for %s: %s", ticker, e)
        _YF_READY = False
        return None

def current_price(ticker: str) -> Optional[float]:
    """Best-effort price (Alpaca → Yahoo)."""
    return _alpaca_price(ticker) or _yf_price(ticker)

# ------------ the insert ------------

def insert_signal_row(
    conn,
    *,
    symbol_id: int,
    ticker: str,
    timeframe: str,
    model: str,
    side: str,         # "buy" | "sell" | "hold"
    strength: float,   # 0..1
    ts: datetime | None = None,
    px: Optional[float] = None,
) -> None:
    """Insert one signal, portable to both schemas (with/without JSONB; with/without px)."""
    ts = ts or datetime.now(timezone.utc)

    use_jsonb = signals_has_jsonb(conn)
    use_px = signals_has_px(conn)

    if use_jsonb and use_px:
        sql = sa.text("""
            INSERT INTO signals
                (ts, symbol_id, ticker, timeframe, model, side, strength, px, signal)
            VALUES
                (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength, :px,
                 jsonb_build_object('side', :side, 'strength', :strength))
        """)
    elif use_jsonb and not use_px:
        sql = sa.text("""
            INSERT INTO signals
                (ts, symbol_id, ticker, timeframe, model, side, strength, signal)
            VALUES
                (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength,
                 jsonb_build_object('side', :side, 'strength', :strength))
        """)
    elif (not use_jsonb) and use_px:
        sql = sa.text("""
            INSERT INTO signals
                (ts, symbol_id, ticker, timeframe, model, side, strength, px)
            VALUES
                (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength, :px)
        """)
    else:
        sql = sa.text("""
            INSERT INTO signals
                (ts, symbol_id, ticker, timeframe, model, side, strength)
            VALUES
                (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength)
        """)

    params = {
        "ts": ts,
        "symbol_id": symbol_id,
        "ticker": ticker,
        "timeframe": timeframe,
        "model": model,
        "side": side,
        "strength": float(strength),
        "px": None if px is None else float(px),
    }
    conn.execute(sql, params)

# ------------ demo driver (kept simple) ------------

def _demo_generate(conn, tickers: list[str]) -> int:
    """Create simple alternating signals for each ticker and record px."""
    inserted = 0
    for t in tickers:
        sid = get_or_create_symbol(conn, t)
        for i in range(12):
            side = "buy" if i % 2 == 0 else "sell"
            strength = round(0.35 + 0.05 * (i % 5), 3)  # 0.35..0.55
            px = current_price(t)  # best effort; may be None if providers blocked
            if px is None:
                log.warning("Price fetch failed for %s; inserting px=NULL", t)
            insert_signal_row(
                conn,
                symbol_id=sid,
                ticker=t,
                timeframe="5m",
                model="ensemble_demo",
                side=side,
                strength=strength,
                px=px,
            )
            inserted += 1
    return inserted

if __name__ == "__main__":
    tickers = [t.strip().upper() for t in os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",") if t.strip()]
    eng = make_engine()
    with eng.begin() as conn:  # transaction
        n = _demo_generate(conn, tickers)
    print(f"✔ inserted {n} signals")

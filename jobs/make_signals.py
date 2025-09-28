# jobs/make_signals.py  (excerpt)
from datetime import datetime, timezone
import os
import sqlalchemy as sa

from lib.db import make_engine  # already in your repo

# ------------ schema helpers ------------

_SIGNALS_HAS_JSONB = None  # cached feature flag

def signals_has_jsonb(conn) -> bool:
    """Return True if 'signals.signal' JSONB column exists (cached)."""
    global _SIGNALS_HAS_JSONB
    if _SIGNALS_HAS_JSONB is not None:
        return _SIGNALS_HAS_JSONB
    q = sa.text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'signals'
          AND column_name  = 'signal'
          AND data_type    = 'jsonb'
        LIMIT 1
    """)
    _SIGNALS_HAS_JSONB = bool(conn.execute(q).scalar())
    return _SIGNALS_HAS_JSONB


def get_or_create_symbol(conn, ticker: str, name: str | None = None) -> int:
    """Return symbols.id for ticker, inserting if needed."""
    q = sa.text("""
        INSERT INTO symbols (ticker, name)
        VALUES (:ticker, COALESCE(:name, :ticker))
        ON CONFLICT (ticker) DO UPDATE
            SET name = COALESCE(EXCLUDED.name, symbols.name)
        RETURNING id
    """)
    return conn.execute(q, {"ticker": ticker, "name": name}).scalar_one()


# ------------ the insert ------------

def insert_signal_row(
    conn,
    *,
    symbol_id: int,
    ticker: str,
    timeframe: str,
    model: str,
    side: str,          # "buy" | "sell"
    strength: float,    # 0..1
    ts: datetime | None = None,
) -> None:
    """Insert one signal, portable to both schemas (with/without JSONB)."""
    ts = ts or datetime.now(timezone.utc)

    if signals_has_jsonb(conn):
        sql = sa.text("""
            INSERT INTO signals
                (ts, symbol_id, ticker, timeframe, model, side, strength, signal)
            VALUES
                (:ts, :symbol_id, :ticker, :timeframe, :model, :side, :strength,
                 jsonb_build_object('side', :side, 'strength', :strength))
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
    }
    conn.execute(sql, params)


# ------------ tiny demo driver (optional) ------------

def _demo_generate(conn, tickers: list[str]) -> int:
    """Create simple alternating signals for each ticker."""
    inserted = 0
    for t in tickers:
        sid = get_or_create_symbol(conn, t)
        # 12 rows per ticker
        for i in range(12):
            side = "buy" if i % 2 == 0 else "sell"
            strength = round(0.35 + 0.05 * (i % 5), 3)  # 0.35..0.55
            insert_signal_row(
                conn,
                symbol_id=sid,
                ticker=t,
                timeframe="1d",
                model="demo_crossover",
                side=side,
                strength=strength,
            )
            inserted += 1
    return inserted


if __name__ == "__main__":
    # Usage: SIGNAL_TICKERS="AAPL,MSFT,SPY" python -m jobs.make_signals
    tickers = [t.strip().upper() for t in os.getenv("SIGNAL_TICKERS", "AAPL,MSFT,SPY").split(",") if t.strip()]
    eng = make_engine()
    with eng.begin() as conn:  # transaction
        n = _demo_generate(conn, tickers)
    print(f"✔ inserted {n} signals")

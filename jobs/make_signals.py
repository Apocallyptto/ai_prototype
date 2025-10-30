# jobs/make_signals.py
import os, logging, psycopg2, time
from datetime import datetime, timezone
from typing import List, Tuple, Optional

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","AAPL,MSFT,SPY").split(",") if s.strip()]
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID","paper")

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET", "")
ALPACA_FEED = os.getenv("ALPACA_FEED", "iex")
USE_ALPACA = bool(ALPACA_KEY and ALPACA_SEC)

# ----- Price fetchers -----
def _price_alpaca(symbol: str) -> Optional[float]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SEC)
        q = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        quote = q[symbol]
        for f in [getattr(quote, "ask_price", None), getattr(quote, "bid_price", None), getattr(quote, "last_price", None)]:
            if f is not None:
                return float(f)
    except Exception as e:
        log.debug(f"alpaca latest quote failed for {symbol}: {e}")
    return None

def _price_yahoo(symbol: str) -> Optional[float]:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.fast_info  # fast path
        p = info.last_price or info.last_trade or info.regular_market_price
        return float(p) if p is not None else None
    except Exception:
        return None

def _price(symbol: str) -> Optional[float]:
    if USE_ALPACA:
        p = _price_alpaca(symbol)
        if p is not None:
            return p
    return _price_yahoo(symbol)

# ----- Your rule/technical ensemble (placeholder) -----
def _rule_signal(symbol: str) -> Tuple[str, float]:
    # Simple example: random-ish placeholder; replace with your real rules
    # Always emit HOLD with medium confidence by default
    return ("hold", 0.0)

# ----- Write to DB -----
INSERT_SQL = """
INSERT INTO public.signals (created_at, symbol, side, strength, source, px, portfolio_id)
VALUES (NOW(), %s, %s, %s, %s, %s, %s)
"""

def _insert_signal(conn, symbol: str, side: str, strength: float, source: str, px: Optional[float]):
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, (symbol, side, strength, source, px, PORTFOLIO_ID))
    conn.commit()

def main():
    with psycopg2.connect(DB_URL) as conn:
        inserted = 0
        for sym in SYMBOLS:
            side, strength = _rule_signal(sym)
            px = _price(sym)  # <-- ensure px is stored
            _insert_signal(conn, sym, side, strength, "rules", px)
            inserted += 1
        log.info(f"✔ inserted {inserted} signals")

if __name__ == "__main__":
    main()

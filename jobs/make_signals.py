# jobs/make_signals.py
import os, logging, psycopg2
from typing import Optional, Tuple

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_signals")

DB_URL = os.getenv("DB_URL","postgresql://postgres:postgres@postgres:5432/trader")
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS","AAPL,MSFT,SPY").split(",") if s.strip()]
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID","paper")

ALPACA_KEY = os.getenv("ALPACA_API_KEY","")
ALPACA_SEC = os.getenv("ALPACA_API_SECRET","")
ALPACA_FEED = os.getenv("ALPACA_FEED","iex")
USE_ALPACA = bool(ALPACA_KEY and ALPACA_SEC)

def _price_alpaca(symbol: str) -> Optional[float]:
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        cli = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SEC)
        q = cli.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        qt = q[symbol]
        for v in (getattr(qt,"ask_price",None), getattr(qt,"bid_price",None), getattr(qt,"last_price",None)):
            if v is not None:
                return float(v)
    except Exception:
        pass
    return None

def _price_yahoo(symbol: str) -> Optional[float]:
    try:
        import yfinance as yf
        p = yf.Ticker(symbol).fast_info.last_price
        return float(p) if p is not None else None
    except Exception:
        return None

def _price(symbol: str) -> Optional[float]:
    if USE_ALPACA:
        p = _price_alpaca(symbol)
        if p is not None: return p
    return _price_yahoo(symbol)

# Replace this with your real technical rules
def _rule_signal(symbol: str) -> Tuple[str,float]:
    # simple nudge: alternate BUY/SELL to avoid HOLDs
    return ("buy", 0.6)  # always a valid side

SQL = """
INSERT INTO public.signals (created_at, symbol, side, strength, source, px, portfolio_id)
VALUES (NOW(), %s, %s, %s, %s, %s, %s)
"""

def _insert(conn, symbol: str, side: str, strength: float, source: str, px: Optional[float]):
    with conn.cursor() as cur:
        cur.execute(SQL, (symbol, side, strength, source, px, PORTFOLIO_ID))
    conn.commit()

def main():
    with psycopg2.connect(DB_URL) as conn:
        n = 0
        for sym in SYMBOLS:
            side, strength = _rule_signal(sym)
            if side == "hold":     # respect DB check constraint
                log.info(f"{sym}: skipping HOLD to satisfy signals_side_check")
                continue
            px = _price(sym)
            _insert(conn, sym, side, strength, "rules", px)
            n += 1
        log.info(f"✔ inserted {n} signals")

if __name__ == "__main__":
    main()

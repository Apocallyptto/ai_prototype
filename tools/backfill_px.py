# tools/backfill_px.py
import os, time, logging
import psycopg2
from decimal import Decimal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_px")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")

# --- price providers (Alpaca first, Yahoo fallback) ---
def _alpaca_last(sym: str):
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        k = os.getenv("ALPACA_API_KEY")
        s = os.getenv("ALPACA_API_SECRET")
        if not (k and s):
            return None
        cli = StockHistoricalDataClient(k, s)
        q = cli.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))
        rec = q.get(sym)
        if not rec:
            return None
        for field in ("ask_price", "bid_price", "last_price"):
            val = getattr(rec, field, None)
            if val:
                return float(val)
    except Exception as e:
        log.warning("Alpaca price fail for %s: %s", sym, e)
    return None

def _yahoo_last(sym: str):
    try:
        import yfinance as yf
        t = yf.Ticker(sym)
        px = t.fast_info.last_price
        if px:
            return float(px)
        # fallback: 1m bars
        df = t.history(period="1d", interval="1m")
        if not df.empty and "Close" in df.columns:
            return float(df["Close"].iloc[-1])
    except Exception as e:
        log.warning("Yahoo price fail for %s: %s", sym, e)
    return None

def _get_px(sym: str):
    return _alpaca_last(sym) or _yahoo_last(sym)

SQL_SELECT_NULLS = """
SELECT id, symbol, created_at
FROM public.signals
WHERE px IS NULL
  AND symbol = ANY(%s)
  AND created_at > NOW() - INTERVAL '2 hours'
ORDER BY created_at DESC
LIMIT 500;
"""

SQL_UPDATE = "UPDATE public.signals SET px = %s WHERE id = %s;"

def main():
    log.info("backfill_px starting; DB=%s", DB_URL)
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        cur.execute(SQL_SELECT_NULLS, (SYMBOLS,))
        rows = cur.fetchall()
        if not rows:
            log.info("No recent signals with NULL px.")
            return
        # fetch one price per symbol, then patch all rows per symbol
        price_cache = {}
        for _, sym, _ in rows:
            if sym not in price_cache:
                price_cache[sym] = _get_px(sym)
                time.sleep(0.3)  # be nice to providers
        patched = 0
        for id_, sym, _ in rows:
            px = price_cache.get(sym)
            if px:
                cur.execute(SQL_UPDATE, (Decimal(str(px)), id_))
                patched += 1
        conn.commit()
        log.info("Backfilled px for %d rows (symbols=%s)", patched, ",".join(SYMBOLS))

if __name__ == "__main__":
    main()

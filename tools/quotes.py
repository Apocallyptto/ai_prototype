# tools/quotes.py
import os, logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("quotes")

def _alpaca_quote(symbol: str) -> Optional[Tuple[float, float]]:
    try:
        # alpaca-py market data client
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        key = os.getenv("ALPACA_API_KEY")
        sec = os.getenv("ALPACA_API_SECRET")
        cli = StockHistoricalDataClient(key, sec)
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        q = cli.get_stock_latest_quote(req)
        # response is mapping {symbol: Quote}
        q = q[symbol]
        bid = float(getattr(q, "bid_price", 0) or 0)
        ask = float(getattr(q, "ask_price", 0) or 0)
        if bid > 0 and ask > 0 and ask >= bid:
            return bid, ask
    except Exception as e:
        log.debug("alpaca quote fallback: %s", e)
    return None

def _yf_quote(symbol: str) -> Optional[Tuple[float, float]]:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", None)
        if info:
            last = float(getattr(info, "last_price", 0) or 0)
            # yfinance often lacks bid/ask intraday; synthesize a tiny spread if needed
            if last > 0:
                spread = max(0.01, min(0.05, last * 0.0005))  # ~5 bps cap
                return last - spread/2, last + spread/2
    except Exception as e:
        log.debug("yfinance quote fallback: %s", e)
    return None

def get_bid_ask_mid(symbol: str) -> Optional[Tuple[float, float, float]]:
    q = _alpaca_quote(symbol) or _yf_quote(symbol)
    if not q:
        return None
    bid, ask = q
    mid = (bid + ask) / 2.0
    return bid, ask, mid

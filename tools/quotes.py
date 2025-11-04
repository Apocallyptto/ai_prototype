"""
Best-effort latest NBBO quote fetch with Alpaca. Falls back gracefully.
Returns (bid, ask, mid) or None if unavailable.
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.common.enums import BaseURL


def _client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY") or ""
    secret = os.getenv("ALPACA_API_SECRET") or ""
    # For paper accounts, market data is still live endpoint (depends on plan).
    return StockHistoricalDataClient(key, secret)


def get_bid_ask_mid(symbol: str) -> Optional[Tuple[float, float, float]]:
    try:
        cli = _client()
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        qmap = cli.get_stock_latest_quote(req)
        q = qmap.get(symbol)
        if not q:
            return None
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2.0
        return bid, ask, mid
    except Exception:
        # If the plan doesn’t include quotes or the call fails, let caller decide.
        return None

# services/position_guard.py
from __future__ import annotations
import os

def _alpaca_client():
    key = os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID")
    sec = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not sec:
        return None
    base = os.getenv("ALPACA_BASE_URL") or os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"
    try:
        from alpaca_trade_api.rest import REST
    except Exception:
        return None
    return REST(key, sec, base_url=base)

def has_same_side_position(symbol: str, side: str) -> bool:
    """
    Returns True if there is already an open position in `symbol` on the same side.
    side: 'buy' means long; 'sell' means short.
    """
    api = _alpaca_client()
    if api is None:
        # No trading API available → cannot detect; be permissive
        return False
    try:
        pos = api.get_position(symbol)
    except Exception:
        return False  # no position
    qty = float(getattr(pos, "qty", 0) or getattr(pos, "qty_available", 0) or 0)
    if qty == 0:
        return False
    # Long if qty>0, short if qty<0
    is_long = qty > 0
    return (side.lower() == "buy" and is_long) or (side.lower() == "sell" and (not is_long))

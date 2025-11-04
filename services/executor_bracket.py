import math
import os
import logging
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.common.exceptions import APIError

from tools.util import pg_connect, market_is_open, retry  # retry is used elsewhere in project
from tools.quotes import get_bid_ask_mid

log = logging.getLogger(__name__)

# ---------- ENV ----------
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))

TP_MULT = float(os.getenv("TP_MULT", "1.5"))   # ×ATR for take-profit
SL_MULT = float(os.getenv("SL_MULT", "1.0"))   # ×ATR for stop-loss

FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
# Treat MIN_QTY as float to allow fractional like 0.01
_default_min_qty = "0.01" if FRACTIONAL else "1"
MIN_QTY = float(os.getenv("MIN_QTY", _default_min_qty))

# If you want to disallow fractional shorts:
ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"


def _round_qty(qty: float) -> float:
    """
    Round quantity for Alpaca:
      - equities fractional allow up to 6 decimals; we keep 2 decimals as a safe default
      - if not fractional, round up to int
    """
    if FRACTIONAL:
        # keep two decimals by default (change to 3/6 if you prefer)
        return round(qty, 2)
    return int(math.ceil(qty))


def _min_qty_ok(side: str, qty: float) -> bool:
    if qty < MIN_QTY:
        return False
    if (side == "sell") and (not FRACTIONAL) and (abs(qty) < 1):
        # non-fractional short must be at least 1 share
        return False
    if (side == "sell") and FRACTIONAL and not ALLOW_FRACTIONAL_SHORTS:
        # you can still short whole shares; just not fractional
        if qty != int(qty):
            return False
    return True


def _side_enum(side: str) -> OrderSide:
    s = side.lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return OrderSide.BUY if s == "buy" else OrderSide.SELL


def _tp_sl_from_atr(limit_price: float, atr: float) -> Tuple[float, float]:
    """
    For BUY:
      TP = limit + ATR*TP_MULT
      SL = limit - ATR*SL_MULT
    For SELL:
      TP = limit - ATR*TP_MULT
      SL = limit + ATR*SL_MULT
    We compute BUY case here; for SELL we flip signs later.
    """
    tp = limit_price + atr * TP_MULT
    sl = max(0.01, limit_price - atr * SL_MULT)
    return tp, sl


def submit_bracket(
    client: TradingClient,
    symbol: str,
    side: str,
    *,
    limit_price: Optional[float] = None,
    qty: Optional[float] = None,
    notional: Optional[float] = None,
    atr: Optional[float] = None,
    extended_hours: bool = False,
) -> str:
    """
    Create a parent limit + OCO children (TP/SL).
    - If limit_price is None, we use current mid quote
    - If qty is None and notional is provided, we derive qty = notional/limit_price
    - Enforces MIN_QTY and fractional policy
    """
    if not ALLOW_AFTER_HOURS and not extended_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0.")

    # price discovery
    bid, ask, mid = get_bid_ask_mid(symbol)
    px = limit_price or mid
    if px is None:
        raise RuntimeError(f"no price for {symbol}")

    # quantity derivation
    if qty is None:
        if notional is None:
            # default: one unit of MIN_QTY
            qty = MIN_QTY
        else:
            qty = notional / px

    qty = abs(_round_qty(qty))
    if not _min_qty_ok(side, qty):
        raise ValueError(f"qty {qty} violates MIN_QTY={MIN_QTY} or short/fractional policy")

    # build TP/SL from ATR (or small price offsets if atr not given)
    if atr and atr > 0:
        tp_buy, sl_buy = _tp_sl_from_atr(px, atr)
    else:
        # fallback: ~0.4% TP and 0.3% SL
        tp_buy = px * 1.004
        sl_buy = px * 0.997

    if side.lower() == "sell":
        # invert for short
        tp = px - (tp_buy - px)
        sl = px + (px - sl_buy)
    else:
        tp = tp_buy
        sl = sl_buy

    # Alpaca bracket
    req = LimitOrderRequest(
        symbol=symbol,
        side=_side_enum(side),
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(px, 4),
        qty=qty,                  # qty always provided (even with fractional)
        order_class="bracket",
        take_profit=TakeProfitRequest(limit_price=round(tp, 4)),
        stop_loss=StopLossRequest(stop_price=round(sl, 4)),
        extended_hours=False,     # regular hours only
    )

    try:
        o = client.submit_order(req)
        log.info("submitted %s %s qty=%s limit=%.4f TP=%.4f SL=%.4f id=%s",
                 symbol, side, qty, px, tp, sl, o.id)
        return o.id
    except APIError as e:
        log.warning("submit failed %s %s: %s", symbol, side, e)
        raise

import math
import os
import logging
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.common.exceptions import APIError

from tools.util import pg_connect, market_is_open, retry  # retry used elsewhere
from tools.quotes import get_bid_ask_mid

log = logging.getLogger(__name__)

# ---------- ENV ----------
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))

TP_MULT = float(os.getenv("TP_MULT", "1.5"))  # ×ATR take-profit
SL_MULT = float(os.getenv("SL_MULT", "1.0"))  # ×ATR stop-loss

FRACTIONAL = os.getenv("FRACTIONAL", "0") == "1"
# IMPORTANT: MIN_QTY is a FLOAT now (supports 0.01 for fractional)
_default_min_qty = "0.01" if FRACTIONAL else "1"
MIN_QTY = float(os.getenv("MIN_QTY", _default_min_qty))

ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"


def _round_qty(qty: float) -> float:
    """Round quantity for Alpaca."""
    if FRACTIONAL:
        return round(qty, 2)  # change precision if you want
    return int(math.ceil(qty))


def _min_qty_ok(side: str, qty: float) -> bool:
    if qty < MIN_QTY:
        return False
    if (side == "sell") and (not FRACTIONAL) and (abs(qty) < 1):
        return False
    if (side == "sell") and FRACTIONAL and not ALLOW_FRACTIONAL_SHORTS:
        if qty != int(qty):
            return False
    return True


def _side_enum(side: str) -> OrderSide:
    s = side.lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return OrderSide.BUY if s == "buy" else OrderSide.SELL


def _tp_sl_from_atr(limit_price: float, atr: float) -> Tuple[float, float]:
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
    Parent limit with OCO TP/SL.
    """
    if not ALLOW_AFTER_HOURS and not extended_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0.")

    bid, ask, mid = get_bid_ask_mid(symbol)
    px = limit_price or mid
    if px is None or px <= 0:
        raise RuntimeError(f"no usable price for {symbol}: bid={bid} ask={ask} mid={mid}")

    if qty is None:
        if notional is None:
            qty = MIN_QTY
        else:
            qty = notional / px

    qty = abs(_round_qty(float(qty)))
    if not _min_qty_ok(side, qty):
        raise ValueError(f"qty {qty} violates MIN_QTY={MIN_QTY} or short/fractional policy")

    if atr and atr > 0:
        tp_buy, sl_buy = _tp_sl_from_atr(px, atr)
    else:
        tp_buy = px * 1.004
        sl_buy = px * 0.997

    if side.lower() == "sell":
        tp = px - (tp_buy - px)
        sl = px + (px - sl_buy)
    else:
        tp = tp_buy
        sl = sl_buy

    req = LimitOrderRequest(
        symbol=symbol,
        side=_side_enum(side),
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(px, 4),
        qty=qty,                       # qty provided explicitly
        order_class="bracket",
        take_profit=TakeProfitRequest(limit_price=round(tp, 4)),
        stop_loss=StopLossRequest(stop_price=round(sl, 4)),
        extended_hours=False,          # regular hours only
    )

    try:
        o = client.submit_order(req)
        log.info("submitted %s %s qty=%s limit=%.4f TP=%.4f SL=%.4f id=%s",
                 symbol, side, qty, px, tp, sl, o.id)
        return o.id
    except APIError as e:
        log.warning("submit failed %s %s: %s", symbol, side, e)
        raise

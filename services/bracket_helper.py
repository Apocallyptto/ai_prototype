import math
import os
import logging
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.common.exceptions import APIError

from tools.util import market_is_open
from tools.quotes import get_bid_ask_mid

log = logging.getLogger(__name__)

# ---------- ENV ----------
ALLOW_AFTER_HOURS       = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
MIN_ACCOUNT_BP_USD      = float(os.getenv("MIN_ACCOUNT_BP_USD", "100"))

TP_MULT                 = float(os.getenv("TP_MULT", "1.5"))  # ×ATR take-profit
SL_MULT                 = float(os.getenv("SL_MULT", "1.0"))  # ×ATR stop-loss

FRACTIONAL              = os.getenv("FRACTIONAL", "0") == "1"
_default_min_qty        = "0.01" if FRACTIONAL else "1"
MIN_QTY                 = float(os.getenv("MIN_QTY", _default_min_qty))

ALLOW_FRACTIONAL_SHORTS = os.getenv("ALLOW_FRACTIONAL_SHORTS", "0") == "1"


def _round_qty(qty: float) -> float:
    """Round quantity for Alpaca."""
    if FRACTIONAL:
        return round(qty, 2)  # fractional to 2 decimals
    return int(max(1, math.floor(qty)))


def _min_qty_ok(side: str, qty: float) -> bool:
    if qty < MIN_QTY:
        return False
    if (side == "sell") and (not FRACTIONAL) and (abs(qty) < 1):
        return False
    if (side == "sell") and FRACTIONAL and not ALLOW_FRACTIONAL_SHORTS:
        # disallow fractional shorts if policy off
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
    Submit a parent LIMIT with OCO TP/SL.
    """
    # RTH gate
    if not ALLOW_AFTER_HOURS and not extended_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0.")

    # Quotes
    bid, ask, mid = get_bid_ask_mid(symbol)
    # guard broken quotes (sometimes ask=0 from free feeds)
    if (bid is None or bid <= 0) and (ask is None or ask <= 0) and (mid is not None and mid > 0):
        bid = ask = mid
    if ask is not None and ask <= 0 and mid and mid > 0:
        ask = mid
    if bid is not None and bid <= 0 and mid and mid > 0:
        bid = mid

    px = float(limit_price) if limit_price else (mid if mid and mid > 0 else bid or ask)
    if px is None or px <= 0:
        raise RuntimeError(f"no usable price for {symbol}: bid={bid} ask={ask} mid={mid}")

    # Quantity
    if qty is None:
        if notional is None:
            qty = MIN_QTY
        else:
            qty = notional / px
    qty = abs(_round_qty(float(qty)))
    if not _min_qty_ok(side, qty):
        raise ValueError(f"qty {qty} violates MIN_QTY={MIN_QTY} or short/fractional policy")

    # TP/SL
    if atr and atr > 0:
        tp_buy, sl_buy = _tp_sl_from_atr(px, atr)
    else:
        # conservative defaults if ATR not given
        tp_buy = px * 1.004
        sl_buy = px * 0.997

    if side.lower() == "sell":
        # mirror for shorts
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
        qty=qty,
        order_class="bracket",
        take_profit=TakeProfitRequest(limit_price=round(tp, 4)),
        stop_loss=StopLossRequest(stop_price=round(sl, 4)),
        # IMPORTANT: honor requested extended hours flag
        extended_hours=bool(extended_hours),
    )

    try:
        o = client.submit_order(req)
        log.info("submitted %s %s qty=%s limit=%.4f TP=%.4f SL=%.4f id=%s ext_hours=%s",
                 symbol, side, qty, px, tp, sl, o.id, extended_hours)
        return o.id
    except APIError as e:
        log.warning("submit failed %s %s: %s", symbol, side, e)
        raise

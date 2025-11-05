import os
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.models import Order

from tools.quotes import get_bid_ask_mid
from tools.atr import get_atr
from tools.util import market_is_open

# --- ENV / switches ---
FRACTIONAL              = os.getenv("FRACTIONAL", "0") == "1"
ALLOW_AFTER_HOURS       = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"
ACCOUNT_FALLBACK_TO_CASH= os.getenv("ACCOUNT_FALLBACK_TO_CASH", "1") == "1"

# MIN_QTY is float when FRACTIONAL=1; int otherwise
if FRACTIONAL:
    MIN_QTY = float(os.getenv("MIN_QTY", "0.01"))
else:
    MIN_QTY = int(os.getenv("MIN_QTY", "1"))

TP_ATR_MULT             = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT             = float(os.getenv("SL_ATR_MULT", "1.0"))
QUOTE_PRICE_SLIPPAGE    = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

MAX_SPREAD_PCT          = float(os.getenv("MAX_SPREAD_PCT", "0.15"))  # %
MAX_SPREAD_ABS          = float(os.getenv("MAX_SPREAD_ABS", "0.06"))  # $


def _coerce_side(side: str) -> OrderSide:
    s = side.lower().strip()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return OrderSide.BUY if s == "buy" else OrderSide.SELL


def _quote_guard_limit(symbol: str, side: str, px_hint: Optional[float]) -> Tuple[float, float, float]:
    """
    Returns (limit_px, bid, ask) using a quote-aware slippage guard.
    Raises ValueError if spread is too wide.
    """
    q = get_bid_ask_mid(symbol)
    if not q:
        # Fallback to px_hint if quotes not available
        if px_hint is None:
            raise ValueError(f"No quotes and no px_hint for {symbol}")
        return round(px_hint, 2), 0.0, 0.0

    bid, ask, mid = q
    spread_abs = max(0.0, ask - bid)
    spread_pct = (spread_abs / mid) * 100.0 if mid > 0 else 999.0

    if (spread_pct > MAX_SPREAD_PCT) or (spread_abs > MAX_SPREAD_ABS):
        raise ValueError(
            f"skip wide spread for {symbol} (bid={bid:.2f} ask={ask:.2f} mid={mid:.2f} "
            f"abs={spread_abs:.4f} pct={spread_pct:.3f}%)"
        )

    slip = QUOTE_PRICE_SLIPPAGE
    if side == "buy":
        limit_px = min(ask + slip, max(ask, px_hint if px_hint else ask))
    else:
        limit_px = max(bid - slip, min(bid, px_hint if px_hint else bid))
    return round(limit_px, 2), bid, ask


def _compute_targets(side: str, entry_ref_px: float, atr: float) -> Tuple[float, float]:
    """
    Returns (tp_px, sl_px) computed from ATR multipliers.
    """
    if side == "buy":
        tp_px = entry_ref_px + TP_ATR_MULT * atr
        sl_px = entry_ref_px - SL_ATR_MULT * atr
    else:
        tp_px = entry_ref_px - TP_ATR_MULT * atr
        sl_px = entry_ref_px + SL_ATR_MULT * atr
    return (round(tp_px, 2), round(sl_px, 2))


def submit_bracket(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    px_hint: Optional[float] = None,
    allow_after_hours: Optional[bool] = None,
) -> str:
    """
    Submit a REGULAR-HOURS-ONLY bracket (entry + TP + SL).
    Alpaca does not allow bracket orders in extended hours.
    """
    if qty < MIN_QTY:
        raise ValueError(f"qty {qty} < MIN_QTY {MIN_QTY}")

    # RTH only
    if allow_after_hours is None:
        allow_after_hours = ALLOW_AFTER_HOURS
    if (not allow_after_hours) and (not market_is_open()):
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0 (brackets are RTH-only).")

    # price + guard
    limit_px, _, _ = _quote_guard_limit(symbol, side, px_hint)

    # ATR for TP/SL
    atr, _last_close = get_atr(symbol)
    tp_px, sl_px = _compute_targets(side, limit_px, atr)

    # Build bracket request
    req = LimitOrderRequest(
        symbol=symbol,
        qty=str(qty) if FRACTIONAL else int(qty),
        side=_coerce_side(side),
        time_in_force=TimeInForce.DAY,
        limit_price=limit_px,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_px),
        stop_loss=StopLossRequest(stop_price=sl_px),
        extended_hours=False,  # force RTH
    )
    o: Order = client.submit_order(req)
    return o.client_order_id or o.id


def submit_simple_entry(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    px_hint: Optional[float] = None,
    extended_hours: bool = False,
) -> str:
    """
    Submit a simple LIMIT entry (no TP/SL). Use this for after-hours testing.
    Your exit manager can attach TP/SL later after the fill.
    """
    if qty < MIN_QTY:
        raise ValueError(f"qty {qty} < MIN_QTY {MIN_QTY}")

    limit_px, _, _ = _quote_guard_limit(symbol, side, px_hint)

    req = LimitOrderRequest(
        symbol=symbol,
        qty=str(qty) if FRACTIONAL else int(qty),
        side=_coerce_side(side),
        time_in_force=TimeInForce.DAY,
        limit_price=limit_px,
        # Not a bracket
        order_class=None,
        extended_hours=extended_hours,
    )
    o: Order = client.submit_order(req)
    return o.client_order_id or o.id

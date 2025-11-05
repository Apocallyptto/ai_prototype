import os
import time
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)

from tools.quotes import get_bid_ask_mid
from tools.atr import get_atr
from tools.util import market_is_open

# -------- ENV / knobs ----------
FRACTIONAL              = os.getenv("FRACTIONAL", "0") == "1"
# if FRACTIONAL=1 we allow floats like 0.01; else int shares
MIN_QTY                 = float(os.getenv("MIN_QTY", "0.01" if FRACTIONAL else "1"))

# Quote guard limits
MAX_SPREAD_PCT          = float(os.getenv("MAX_SPREAD_PCT", "0.15"))   # percent
MAX_SPREAD_ABS          = float(os.getenv("MAX_SPREAD_ABS", "0.12"))   # dollars (relaxed a bit from 0.06)
QUOTE_PRICE_SLIPPAGE    = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

# ATR multipliers for exits
TP_ATR_MULT             = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT             = float(os.getenv("SL_ATR_MULT", "1.0"))

# Spread wait (auto-retry) before giving up
SPREAD_WAIT_SECONDS     = float(os.getenv("SPREAD_WAIT_SECONDS", "6"))     # total wait
SPREAD_POLL_MS          = float(os.getenv("SPREAD_POLL_MS", "250"))        # poll cadence

# -------- utils ----------
def _qt(x: float, places: int = 2) -> float:
    """Hard snap to N decimals. Adds tiny epsilon to avoid 2.1999999 artifacts."""
    return round(float(x) + 1e-9, places)

def _parse_side(side: str) -> OrderSide:
    s = side.lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return OrderSide.BUY if s == "buy" else OrderSide.SELL

# -------- quote-aware limit ----------
def _quote_guard_limit(symbol: str, side: str, px_hint: Optional[float]) -> Tuple[float, float, float]:
    """
    Returns (limit_px, bid, ask) using a quote-aware slippage guard.
    Raises ValueError if spread too wide.
    """
    q = get_bid_ask_mid(symbol)
    if not q:
        if px_hint is None:
            raise ValueError(f"No quotes and no px_hint for {symbol}")
        # fallback: just round hint
        return _qt(px_hint), 0.0, 0.0

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
        raw_limit = min(ask + slip, max(ask, px_hint if px_hint else ask))
    else:
        raw_limit = max(bid - slip, min(bid, px_hint if px_hint else bid))

    limit_px = _qt(raw_limit, 2)  # pennies only
    return limit_px, bid, ask

def _quote_guard_limit_wait(symbol: str, side: str, px_hint: Optional[float]) -> Tuple[float, float, float]:
    """
    Retry _quote_guard_limit() for up to SPREAD_WAIT_SECONDS if spread is temporarily too wide.
    """
    deadline = time.time() + SPREAD_WAIT_SECONDS
    last_err = None
    while True:
        try:
            return _quote_guard_limit(symbol, side, px_hint)
        except ValueError as e:
            msg = str(e)
            if "skip wide spread" not in msg:
                # other error -> bubble up
                raise
            last_err = e
            if time.time() >= deadline:
                raise last_err
            time.sleep(SPREAD_POLL_MS / 1000.0)

# -------- main submit ----------
def submit_bracket(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    px_hint: Optional[float] = None,
    allow_after_hours: bool = False,
) -> str:
    """
    Submit a bracket order with quote-aware entry, ATR-based TP/SL, and penny rounding.
    Returns client_order_id.
    """
    # Regular-hours only (bracket orders cannot be extended-hours on Alpaca)
    if not allow_after_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0.")

    if qty < MIN_QTY:
        raise ValueError(f"qty {qty} < MIN_QTY {MIN_QTY}")

    alp_side = _parse_side(side)

    # 1) Entry price from quotes (with a small retry for fair spread)
    limit_px, bid, ask = _quote_guard_limit_wait(symbol, side, px_hint)

    # 2) ATR for exits
    try:
        atr_val, ref_px = get_atr(symbol)   # returns (atr, last_close_px)
    except Exception:
        # very defensive fallback if ATR path has an issue
        atr_val, ref_px = 0.50, limit_px

    # Entry reference for calculating exits:
    entry_ref = px_hint if px_hint is not None else ref_px

    # Take-profit / Stop-loss (ATR-based)
    if side.lower() == "buy":
        tp_px = entry_ref + TP_ATR_MULT * atr_val
        sl_px = entry_ref - SL_ATR_MULT * atr_val
    else:
        tp_px = entry_ref - TP_ATR_MULT * atr_val
        sl_px = entry_ref + SL_ATR_MULT * atr_val

    tp_px = _qt(tp_px, 2)
    sl_px = _qt(sl_px, 2)

    # 3) Build the request
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty if FRACTIONAL else int(qty),
        side=alp_side,
        time_in_force=TimeInForce.DAY,
        limit_price=_qt(limit_px, 2),
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=_qt(tp_px, 2)),
        stop_loss=StopLossRequest(stop_price=_qt(sl_px, 2)),
        extended_hours=False,  # brackets: regular hours only
    )

    # 4) Submit
    o = client.submit_order(req)
    return o.client_order_id

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

# ======================
# ENV / knobs
# ======================
FRACTIONAL              = os.getenv("FRACTIONAL", "0") == "1"
MIN_QTY                 = float(os.getenv("MIN_QTY", "0.01" if FRACTIONAL else "1"))

MAX_SPREAD_PCT          = float(os.getenv("MAX_SPREAD_PCT", "0.15"))   # percent
MAX_SPREAD_ABS          = float(os.getenv("MAX_SPREAD_ABS", "0.12"))   # dollars
QUOTE_PRICE_SLIPPAGE    = float(os.getenv("QUOTE_PRICE_SLIPPAGE", "0.02"))

TP_ATR_MULT_DEFAULT     = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT_DEFAULT     = float(os.getenv("SL_ATR_MULT", "1.0"))

SPREAD_WAIT_SECONDS     = float(os.getenv("SPREAD_WAIT_SECONDS", "6"))
SPREAD_POLL_MS          = float(os.getenv("SPREAD_POLL_MS", "250"))

# ======================
# Helpers
# ======================
def _qt(x: float, places: int = 2) -> float:
    """Snap to cents (or given decimal places) with tiny epsilon to avoid fp artifacts."""
    return round(float(x) + 1e-9, places)

def _parse_side(side: str) -> OrderSide:
    s = side.lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    return OrderSide.BUY if s == "buy" else OrderSide.SELL

def _quote_guard_limit(symbol: str, side: str, px_hint: Optional[float]) -> Tuple[float, float, float]:
    """
    Returns (limit_px, bid, ask) using a quote-aware slippage guard.
    Raises ValueError if spread too wide or no quotes and no hint.
    """
    q = get_bid_ask_mid(symbol)
    if not q:
        if px_hint is None:
            raise ValueError(f"No quotes and no px_hint for {symbol}")
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

    return _qt(raw_limit, 2), bid, ask

def _quote_guard_limit_wait(symbol: str, side: str, px_hint: Optional[float]) -> Tuple[float, float, float]:
    """Retry for transient wide spreads up to SPREAD_WAIT_SECONDS."""
    deadline = time.time() + SPREAD_WAIT_SECONDS
    last_err = None
    while True:
        try:
            return _quote_guard_limit(symbol, side, px_hint)
        except ValueError as e:
            msg = str(e)
            if "skip wide spread" not in msg:
                raise
            last_err = e
            if time.time() >= deadline:
                raise last_err
            time.sleep(SPREAD_POLL_MS / 1000.0)

# ======================
# Public API
# ======================
def submit_bracket(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    px_hint: Optional[float] = None,
    allow_after_hours: bool = False,
    tp_mult: Optional[float] = None,
    sl_mult: Optional[float] = None,
) -> str:
    """
    Submit a bracket order (regular hours only) with quote-aware entry and ATR TP/SL.
    Returns client_order_id.
    """
    # Alpaca: bracket orders are regular-hours only
    if not allow_after_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0.")

    if qty < MIN_QTY:
        raise ValueError(f"qty {qty} < MIN_QTY {MIN_QTY}")

    alp_side = _parse_side(side)

    # Entry price with quote spread guard + small retry
    limit_px, bid, ask = _quote_guard_limit_wait(symbol, side, px_hint)

    # ATR-based exits (defensive fallback)
    try:
        atr_val, ref_px = get_atr(symbol)   # (atr, last_close)
    except Exception:
        atr_val, ref_px = 0.50, limit_px

    entry_ref = px_hint if px_hint is not None else ref_px
    tp_k = TP_ATR_MULT_DEFAULT if tp_mult is None else float(tp_mult)
    sl_k = SL_ATR_MULT_DEFAULT if sl_mult is None else float(sl_mult)

    if side.lower() == "buy":
        tp_px = entry_ref + tp_k * atr_val
        sl_px = entry_ref - sl_k * atr_val
    else:
        tp_px = entry_ref - tp_k * atr_val
        sl_px = entry_ref + sl_k * atr_val

    tp_px = _qt(tp_px, 2)
    sl_px = _qt(sl_px, 2)

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty if FRACTIONAL else int(qty),
        side=alp_side,
        time_in_force=TimeInForce.DAY,
        limit_price=_qt(limit_px, 2),
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp_px),
        stop_loss=StopLossRequest(stop_price=sl_px),
        extended_hours=False,   # brackets cannot be extended hours
    )
    o = client.submit_order(req)
    return o.client_order_id


def submit_simple_entry(
    client: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    px_hint: Optional[float] = None,
    allow_after_hours: bool = False,
) -> str:
    """
    Submit a simple LIMIT order (no OCO). Useful for testing fills or after-hours checks.
    Returns client_order_id.
    """
    if not allow_after_hours and not market_is_open():
        raise RuntimeError("Market is closed and ALLOW_AFTER_HOURS=0 for simple order.")

    if qty < MIN_QTY:
        raise ValueError(f"qty {qty} < MIN_QTY {MIN_QTY}")

    alp_side = _parse_side(side)
    limit_px, _, _ = _quote_guard_limit_wait(symbol, side, px_hint)

    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty if FRACTIONAL else int(qty),
        side=alp_side,
        time_in_force=TimeInForce.DAY,
        limit_price=_qt(limit_px, 2),
        extended_hours=bool(allow_after_hours),
    )
    o = client.submit_order(req)
    return o.client_order_id

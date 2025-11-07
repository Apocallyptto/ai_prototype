import os
from typing import Optional, Tuple, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.common.exceptions import APIError

# Use Alpaca data client for robust quoting (works inside container)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
USE_BRACKET_THRESHOLD = float(os.getenv("USE_BRACKET_THRESHOLD", "1.0"))  # >=1 -> native bracket
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "1") == "1"             # simple orders only
AH_LIMIT_OFFSET = float(os.getenv("AH_LIMIT_OFFSET", "0.02"))               # $ cushion AH if we must synthesize
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "1.00"))                     # broker min notional
BPS_OFFSET = float(os.getenv("AH_BPS_OFFSET", "4"))                         # optional % cushion if quotes exist (4 bps = 0.04%)

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)

def _qqty(q: float) -> float:
    return round(max(0.0, float(q)), 3)

def _open_orders(trading: TradingClient, symbol: str) -> List:
    return trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol]))

def _cancel_opposite(trading: TradingClient, symbol: str, new_side: OrderSide):
    opp = OrderSide.SELL if new_side == OrderSide.BUY else OrderSide.BUY
    for o in _open_orders(trading, symbol):
        try:
            if getattr(o, "side", None) == opp:
                trading.cancel_order_by_id(o.id)
        except Exception:
            pass

def _is_rth(trading: TradingClient) -> bool:
    try:
        clk = trading.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return True

def _latest_mid(dc: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    """Best-effort mid; fall back to last trade; None if nothing."""
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    try:
        t = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
        if t and t.price:
            return float(t.price)
    except Exception:
        pass
    return None

def _ensure_min_notional(price: float, qty: float) -> float:
    """If price*qty < MIN_NOTIONAL, bump price just enough to satisfy broker floor."""
    notional = price * qty
    if notional >= MIN_NOTIONAL:
        return price
    needed = MIN_NOTIONAL / max(qty, 1e-9)
    return max(price, _qt(needed, 2))

def _tp_sl_from_ref(is_buy: bool, ref_px: float, atr: Optional[float]):
    a = atr if atr is not None else 0.50
    if is_buy:
        tp = _qt(ref_px + TP_ATR_MULT * a, 2)
        sl = _qt(max(0.01, ref_px - SL_ATR_MULT * a), 2)
        if tp <= sl: tp, sl = sl + 0.02, tp - 0.02
    else:
        tp = _qt(ref_px - TP_ATR_MULT * a, 2)
        sl = _qt(ref_px + SL_ATR_MULT * a, 2)
        if tp >= sl: tp, sl = sl - 0.02, tp + 0.02
    return tp, sl

def place_entry(
    trading: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    use_limit: bool = False,
    limit_price: Optional[float] = None,
):
    """
    - Cancels opposite-side OPEN orders (prevents wash trade 403).
    - qty >= USE_BRACKET_THRESHOLD -> native bracket (RTH only).
    - qty <  threshold -> simple order.
       - RTH: MARKET/ LIMIT per user.
       - AH:  force LIMIT + extended_hours=True using robust Alpaca quotes.
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    qty = _qqty(qty)

    # Pre-cancel opposite side
    _cancel_opposite(trading, symbol, side_enum)

    is_rth = _is_rth(trading)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # Whole-share path (native bracket) — valid only RTH
    if qty >= USE_BRACKET_THRESHOLD:
        # For bracket we need TP/SL ref. Use latest mid if available.
        mid = _latest_mid(dc, symbol)
        ref = limit_price if (use_limit and limit_price is not None) else (mid if mid is not None else 0.0)

        # If not RTH, degrade to AH-safe simple LIMIT (bracket not allowed), priced near mid with BPS cushion.
        if not is_rth:
            px = limit_price if (use_limit and limit_price is not None) else (
                (_qt(ref * (1 + (BPS_OFFSET/10000.0)), 2) if side_enum == OrderSide.BUY else
                 _qt(ref * (1 - (BPS_OFFSET/10000.0)), 2))
            )
            if ref == 0.0 or px <= 0.0:
                px = _qt(AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else max(0.01, AH_LIMIT_OFFSET), 2)
            px = _ensure_min_notional(px, qty)
            req2 = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                     time_in_force=TimeInForce.DAY, qty=int(qty),
                                     limit_price=px, extended_hours=ALLOW_AFTER_HOURS)
            return trading.submit_order(req2)

        # RTH bracket
        # Compute TP/SL from ref (skip ATR to keep router lean; exits will still be safe).
        tp, sl = _tp_sl_from_ref(is_buy=(side_enum == OrderSide.BUY), ref_px=ref, atr=None)
        if use_limit and limit_price is not None:
            req = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                    time_in_force=TimeInForce.DAY, qty=int(qty),
                                    limit_price=_qt(limit_price, 2),
                                    order_class="bracket", extended_hours=False)
        else:
            req = MarketOrderRequest(symbol=symbol, side=side_enum, type=OrderType.MARKET,
                                     time_in_force=TimeInForce.DAY, qty=int(qty),
                                     order_class="bracket", extended_hours=False)
        return trading.submit_order(req, take_profit={"limit_price": tp}, stop_loss={"stop_price": sl})

    # Fractional path — simple order (synthetic exits attach)
    if not is_rth:
        mid = _latest_mid(dc, symbol)
        px = limit_price if (use_limit and limit_price is not None) else (
            (_qt(mid * (1 + (BPS_OFFSET/10000.0)), 2) if (mid and side_enum == OrderSide.BUY) else
             _qt(mid * (1 - (BPS_OFFSET/10000.0)), 2) if mid and side_enum == OrderSide.SELL
             else None)
        )
        if px is None or px <= 0.0:
            # last fallback: fixed-dollar cushion if quotes fail
            px = _qt(AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else max(0.01, AH_LIMIT_OFFSET), 2)
        px = _ensure_min_notional(px, qty)
        req = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                time_in_force=TimeInForce.DAY, qty=qty,
                                limit_price=px, extended_hours=ALLOW_AFTER_HOURS)
        return trading.submit_order(req)

    # RTH fractional
    if use_limit and limit_price is not None:
        req = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                time_in_force=TimeInForce.DAY, qty=qty,
                                limit_price=_qt(limit_price, 2), extended_hours=False)
        return trading.submit_order(req)
    else:
        req = MarketOrderRequest(symbol=symbol, side=side_enum, type=OrderType.MARKET,
                                 time_in_force=TimeInForce.DAY, qty=qty, extended_hours=False)
        return trading.submit_order(req)

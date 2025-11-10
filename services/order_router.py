import os
from typing import Optional, Tuple, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.common.exceptions import APIError

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
USE_BRACKET_THRESHOLD = float(os.getenv("USE_BRACKET_THRESHOLD", "1.0"))  # >=1 -> native bracket
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "1") == "1"
AH_LIMIT_OFFSET = float(os.getenv("AH_LIMIT_OFFSET", "0.02"))
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "1.00"))
BPS_OFFSET = float(os.getenv("AH_BPS_OFFSET", "4"))  # 4 bps = 0.04%

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

# --- dedupe knobs ---
DEDUPE_SAME_SIDE = os.getenv("DEDUPE_SAME_SIDE", "1") == "1"
DEDUPE_PRICE_TICKS = float(os.getenv("DEDUPE_PRICE_TICKS", "0.02"))  # treat as same if |px-new| <= this
DEDUPE_REPLACE = os.getenv("DEDUPE_REPLACE", "0") == "1"             # if true, cancel & replace; else, just skip


def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)


def _qqty(q: float) -> float:
    return round(max(0.0, float(q)), 3)


def _open_orders(trading: TradingClient, symbol: str):
    return trading.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
            symbols=[symbol],
        )
    )


def _cancel_side(trading: TradingClient, symbol: str, side: OrderSide):
    for o in _open_orders(trading, symbol):
        try:
            if getattr(o, "side", None) == side:
                trading.cancel_order_by_id(o.id)
        except Exception:
            pass


def _cancel_opposite(trading: TradingClient, symbol: str, new_side: OrderSide):
    opp = OrderSide.SELL if new_side == OrderSide.BUY else OrderSide.BUY
    _cancel_side(trading, symbol, opp)


def _is_rth(trading: TradingClient) -> bool:
    try:
        clk = trading.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return True


def _latest_mid(dc: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid and ask and bid > 0 and ask > 0:
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
        if tp <= sl:
            tp, sl = sl + 0.02, max(0.01, sl - 0.02)
    else:
        tp = _qt(ref_px - TP_ATR_MULT * a, 2)
        sl = _qt(ref_px + SL_ATR_MULT * a, 2)
        if tp >= sl:
            tp, sl = tp - 0.02, sl + 0.02
    return tp, sl


def _dedupe_same_side(
    trading: TradingClient,
    symbol: str,
    side: OrderSide,
    candidate_px: Optional[float],
    is_limit: bool,
) -> bool:
    """
    Returns True if we should SKIP placing (duplicate exists),
    or cancels & returns False (if DEDUPE_REPLACE).
    """
    if not DEDUPE_SAME_SIDE:
        return False
    od = _open_orders(trading, symbol)
    for o in od:
        try:
            if getattr(o, "side", None) != side:
                continue
            if is_limit and getattr(o, "type", None) == OrderType.LIMIT and candidate_px is not None:
                opx = float(getattr(o, "limit_price", 0) or 0)
                if abs(opx - candidate_px) <= DEDUPE_PRICE_TICKS:
                    # same-side, near-same price
                    if DEDUPE_REPLACE:
                        trading.cancel_order_by_id(o.id)
                        return False  # allow new placement
                    else:
                        # skip new placement
                        return True
            else:
                # MARKET or unknown price — treat as duplicate, skip to be safe
                if not is_limit:
                    return True
        except Exception:
            pass
    return False


def place_entry(
    trading: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    use_limit: bool = False,
    limit_price: Optional[float] = None,
):
    """
    Safe entry router:
      - cancels opposite
      - dedupes same-side open entries (prevents multiple pending BUYS)
      - AH fractional -> LIMIT with min-notional guard
      - whole-share RTH -> native bracket else degrade to AH LIMIT
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    qty = _qqty(qty)

    # 1) wash-trade guard: cancel opposite side
    _cancel_opposite(trading, symbol, side_enum)

    is_rth = _is_rth(trading)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

    # === Whole-share path (native bracket in RTH) ===
    if qty >= USE_BRACKET_THRESHOLD:
        mid = _latest_mid(dc, symbol)
        ref = limit_price if (use_limit and limit_price is not None) else (mid if mid is not None else 0.0)

        if not is_rth:
            px = limit_price if (use_limit and limit_price is not None) else (
                (_qt(ref * (1 + (BPS_OFFSET/10000.0)), 2) if side_enum == OrderSide.BUY else
                 _qt(ref * (1 - (BPS_OFFSET/10000.0)), 2))
            )
            if ref == 0.0 or px <= 0.0:
                px = _qt(AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else max(0.01, AH_LIMIT_OFFSET), 2)

            px = _ensure_min_notional(px, qty)

            # de-dupe same-side (AH LIMIT)
            if _dedupe_same_side(trading, symbol, side_enum, px, is_limit=True):
                return None

            req2 = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                     time_in_force=TimeInForce.DAY, qty=int(qty),
                                     limit_price=px, extended_hours=ALLOW_AFTER_HOURS)
            return trading.submit_order(req2)

        # RTH bracket
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

        # Market/Limit bracket: still fine to place; no dedupe for bracket here (signals should gate)
        return trading.submit_order(req, take_profit={"limit_price": tp}, stop_loss={"stop_price": sl})

    # === Fractional path (simple order; exits are synthetic) ===
    if not is_rth:
        mid = _latest_mid(dc, symbol)
        px = limit_price if (use_limit and limit_price is not None) else (
            (_qt(mid * (1 + (BPS_OFFSET/10000.0)), 2) if (mid and side_enum == OrderSide.BUY) else
             _qt(mid * (1 - (BPS_OFFSET/10000.0)), 2) if mid and side_enum == OrderSide.SELL
             else None)
        )
        if px is None or px <= 0.0:
            px = _qt(AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else max(0.01, AH_LIMIT_OFFSET), 2)
        px = _ensure_min_notional(px, qty)

        # de-dupe same-side (AH LIMIT)
        if _dedupe_same_side(trading, symbol, side_enum, px, is_limit=True):
            return None

        req = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                time_in_force=TimeInForce.DAY, qty=qty,
                                limit_price=px, extended_hours=ALLOW_AFTER_HOURS)
        return trading.submit_order(req)

    # RTH fractional
    if use_limit and limit_price is not None:
        # de-dupe same-side (RTH LIMIT)
        px = _qt(limit_price, 2)
        if _dedupe_same_side(trading, symbol, side_enum, px, is_limit=True):
            return None

        req = LimitOrderRequest(symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                                time_in_force=TimeInForce.DAY, qty=qty,
                                limit_price=px, extended_hours=False)
        return trading.submit_order(req)
    else:
        # MARKET RTH — dedupe by skipping if there’s already any same-side open MARKET (rare)
        if _dedupe_same_side(trading, symbol, side_enum, None, is_limit=False):
            return None

        req = MarketOrderRequest(symbol=symbol, side=side_enum, type=OrderType.MARKET,
                                 time_in_force=TimeInForce.DAY, qty=qty, extended_hours=False)
        return trading.submit_order(req)

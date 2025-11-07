import os
from typing import Optional, Tuple, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
from alpaca.common.exceptions import APIError

try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
USE_BRACKET_THRESHOLD = float(os.getenv("USE_BRACKET_THRESHOLD", "1.0"))  # >=1 -> native bracket
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "1") == "1"             # simple orders only
AH_LIMIT_OFFSET = float(os.getenv("AH_LIMIT_OFFSET", "0.02"))               # $0.02 price cushion AH

def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)

def _qqty(q: float) -> float:
    return round(max(0.0, float(q)), 3)

def _atr(symbol: str, lookback_days: int = 30, period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    if yf is not None and pd is not None:
        try:
            df = yf.download(symbol, period=f"{lookback_days}d", interval="1d",
                             progress=False, auto_adjust=False, threads=False)
            if df is not None and not df.empty and {"High","Low","Close"}.issubset(df.columns):
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                close = df["Close"].astype(float)
                prev_close = close.shift(1)
                tr1 = (high - low).abs()
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
                return float(atr), float(close.iloc[-1])
        except Exception:
            pass
    return None, None

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

def _open_orders(trading: TradingClient, symbol: str) -> List:
    return trading.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol]))

def _cancel_opposite(trading: TradingClient, symbol: str, new_side: OrderSide):
    opp = OrderSide.SELL if new_side == OrderSide.BUY else OrderSide.BUY
    open_os = _open_orders(trading, symbol)
    for o in open_os:
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

def _latest_quote_price(trading: TradingClient, symbol: str) -> Tuple[Optional[float], Optional[float]]:
    # We avoid importing data client here to keep router light; rely on last close if needed via ATR helper.
    # For AH limit fallback, we can use ATR-ref (close) if quotes aren't available through this light path.
    _, close = _atr(symbol)
    # As a conservative fallback, return (close, close)
    return close, close

def place_entry(
    trading: TradingClient,
    symbol: str,
    side: str,
    qty: float,
    use_limit: bool = False,
    limit_price: Optional[float] = None,
):
    """
    Routes entry:
      - Cancels opposite-side OPEN orders (prevents wash trades).
      - qty >= USE_BRACKET_THRESHOLD -> native bracket (RTH only, extended_hours=False).
      - qty <  threshold -> simple order.
         - During RTH: MARKET or LIMIT per user choice.
         - After-hours: FORCE LIMIT with extended_hours=True (never MARKET AH).
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    qty = _qqty(qty)

    # 1) Pre-cancel opposite-side open orders for this symbol (prevents wash-trade 403)
    _cancel_opposite(trading, symbol, side_enum)

    # 2) Decide path
    is_rth = _is_rth(trading)

    if qty >= USE_BRACKET_THRESHOLD:
        # Native bracket — only valid RTH
        atr, close = _atr(symbol)
        ref = limit_price if (use_limit and limit_price is not None) else (close if close is not None else 0.0)
        tp, sl = _tp_sl_from_ref(is_buy=(side_enum == OrderSide.BUY), ref_px=ref, atr=atr)

        if use_limit and limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY, qty=int(qty),
                limit_price=_qt(limit_price, 2),
                order_class="bracket", extended_hours=False,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, side=side_enum, type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY, qty=int(qty),
                order_class="bracket", extended_hours=False,
            )
        if not is_rth:
            # If outside RTH, convert to LIMIT near ref and submit as simple (synthetic exits will attach)
            px = _qt(ref + (AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else -AH_LIMIT_OFFSET), 2)
            req2 = LimitOrderRequest(
                symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY, qty=int(qty),
                limit_price=px, extended_hours=ALLOW_AFTER_HOURS,
            )
            return trading.submit_order(req2)

        # pass child legs at submit time
        return trading.submit_order(req, take_profit={"limit_price": tp}, stop_loss={"stop_price": sl})

    # Fractional simple
    if not is_rth:
        # Force LIMIT AH (never market AH)
        ref, _ = _latest_quote_price(trading, symbol)
        if ref is None:
            ref = limit_price if limit_price is not None else 0.0
        px = (limit_price if (use_limit and limit_price is not None)
              else _qt(ref + (AH_LIMIT_OFFSET if side_enum == OrderSide.BUY else -AH_LIMIT_OFFSET), 2))
        req = LimitOrderRequest(
            symbol=symbol, side=side_enum, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, qty=qty,
            limit_price=_qt(px, 2), extended_hours=ALLOW_AFTER_HOURS,
        )
        return trading.submit_order(req)

    # RTH fractional → user’s choice (market/limit)
    if use_limit and limit_price is not None:
        req = LimitOrderRequest(
            symbol=symbol, side=side_enum, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, qty=qty,
            limit_price=_qt(limit_price, 2), extended_hours=False,
        )
        return trading.submit_order(req)
    else:
        req = MarketOrderRequest(
            symbol=symbol, side=side_enum, type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY, qty=qty, extended_hours=False,
        )
        return trading.submit_order(req)

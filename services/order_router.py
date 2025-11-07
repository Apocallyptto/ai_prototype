import os
import math
from typing import Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.common.exceptions import APIError

try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
USE_BRACKET_THRESHOLD = float(os.getenv("USE_BRACKET_THRESHOLD", "1.0"))  # >=1 share -> bracket
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "0") == "1"             # for simple orders only

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
    else:
        tp = _qt(ref_px - TP_ATR_MULT * a, 2)
        sl = _qt(ref_px + SL_ATR_MULT * a, 2)
    if is_buy and tp <= sl:
        tp, sl = sl + 0.02, tp - 0.02
    if (not is_buy) and tp >= sl:
        tp, sl = sl - 0.02, tp + 0.02
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
    Routes entry:
      - qty >= USE_BRACKET_THRESHOLD -> native bracket (RTH only)
      - qty <  threshold -> simple order, exits via synthetic monitor

    Returns order object.
    """
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    qty = _qqty(qty)

    # Decide path
    if qty >= USE_BRACKET_THRESHOLD:
        # Native bracket (RTH only; extended_hours=False)
        atr, close = _atr(symbol)
        ref = limit_price if (use_limit and limit_price is not None) else (close if close is not None else 0.0)
        tp, sl = _tp_sl_from_ref(is_buy=(side_enum == OrderSide.BUY), ref_px=ref, atr=atr)

        if use_limit and limit_price is not None:
            req = LimitOrderRequest(
                symbol=symbol, side=side_enum, type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY, qty=int(qty),  # whole shares
                limit_price=_qt(limit_price, 2),
                order_class="bracket",
                extended_hours=False,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol, side=side_enum, type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY, qty=int(qty),
                order_class="bracket",
                extended_hours=False,
            )

        # SDK requires passing child legs at submit time:
        try:
            o = trading.submit_order(
                req,
                take_profit={"limit_price": tp},
                stop_loss={"stop_price": sl},
            )
            return o
        except APIError as e:
            raise

    # Fractional simple → synthetic exits will attach
    if use_limit and limit_price is not None:
        req = LimitOrderRequest(
            symbol=symbol, side=side_enum, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, qty=qty,
            limit_price=_qt(limit_price, 2),
            extended_hours=ALLOW_AFTER_HOURS,
        )
    else:
        req = MarketOrderRequest(
            symbol=symbol, side=side_enum, type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY, qty=qty,
            extended_hours=ALLOW_AFTER_HOURS,
        )
    return trading.submit_order(req)

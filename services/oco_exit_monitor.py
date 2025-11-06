import os
import time
import logging
from typing import Optional, Tuple

import math

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import (
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

try:
    import pandas as pd
    import yfinance as yf
except Exception:
    yf = None
    pd = None

log = logging.getLogger("oco_exit_monitor")
logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"))

TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
EXIT_MONITOR_POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def _qt(x: float, places: int = 2) -> float:
    return round(float(x) + 1e-9, places)

def _latest_bid_ask_mid(data_client: StockHistoricalDataClient, symbol: str):
    try:
        q = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
        qsym = q[symbol]
        bid = float(qsym.bid_price) if qsym.bid_price is not None else None
        ask = float(qsym.ask_price) if qsym.ask_price is not None else None
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        return bid, ask, mid
    except Exception:
        try:
            t = data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))[symbol]
            last = float(t.price)
            return None, None, last
        except Exception:
            return None, None, None

def _atr(symbol: str, lookback_days: int = 30, period: int = 14):
    if yf is not None and pd is not None:
        try:
            df = yf.download(symbol, period=f"{lookback_days}d", interval="1d", progress=False, auto_adjust=False, threads=False)
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
                ref = float(close.iloc[-1])
                return float(atr) if not math.isnan(atr) else None, ref
        except Exception:
            pass
    return None, None

def _derive_tp_sl(side: str, ref_px: float, atr_val: Optional[float]) -> Tuple[float, float]:
    if atr_val is None:
        atr_val = 0.50
    if side == "long":
        tp = _qt(ref_px + TP_ATR_MULT * atr_val, 2)   # higher
        sl = _qt(max(0.01, ref_px - SL_ATR_MULT * atr_val), 2)  # lower
    else:
        tp = _qt(ref_px - TP_ATR_MULT * atr_val, 2)   # lower
        sl = _qt(ref_px + SL_ATR_MULT * atr_val, 2)   # higher
    return tp, sl

def _has_any_children(trading: TradingClient, symbol: str) -> bool:
    try:
        open_orders = trading.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol])
        )
        for o in open_orders:
            oc = getattr(o, "order_class", None)
            if oc in ("bracket", "oco"):
                return True
            if getattr(o, "take_profit", None) or getattr(o, "stop_loss", None):
                return True
        return False
    except Exception:
        return False

def _submit_oco_exit(trading: TradingClient, symbol: str, long_side: bool, qty: float, tp: float, sl: float):
    """
    Submit a TRUE OCO exit (no parent). For a long position -> side=SELL with TP>SL.
    For a short position -> side=BUY with TP<SL. This matches Alpaca's OCO spec.
    """
    side = OrderSide.SELL if long_side else OrderSide.BUY

    # For OCO, API requires type=LIMIT and stop_loss.stop_price present.
    req = LimitOrderRequest(
        symbol=symbol,
        side=side,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        order_class="oco",
        take_profit=TakeProfitRequest(limit_price=_qt(tp, 2)),
        stop_loss=StopLossRequest(stop_price=_qt(sl, 2)),
        extended_hours=False,  # exits are RTH-only per Alpaca docs
    )
    if DRY_RUN:
        log.info("[DRY_RUN] would submit OCO exit %s qty=%.4f | TP=%.2f SL=%.2f", symbol, qty, tp, sl)
        return
    o = trading.submit_order(req)
    log.info("OCO exits attached -> %s oco_parent_id=%s", symbol, o.id)

def run_once(trading: TradingClient, data_client: StockHistoricalDataClient):
    positions = trading.get_all_positions()
    if not positions:
        log.info("no positions")
        return

    for p in positions:
        symbol = p.symbol
        long_side = float(p.qty) > 0
        qty = abs(float(p.qty))
        avg_entry = float(p.avg_entry_price)

        if _has_any_children(trading, symbol):
            log.info("%s: exits already present", symbol)
            continue

        bid, ask, mid = _latest_bid_ask_mid(data_client, symbol)
        ref_px = mid if mid is not None else avg_entry

        atr_val, close_ref = _atr(symbol)
        ref_for_tp_sl = ref_px if ref_px is not None else (close_ref if close_ref else avg_entry)

        tp, sl = _derive_tp_sl("long" if long_side else "short", ref_for_tp_sl, atr_val)

        # For long: ensure TP > SL; for short: TP < SL (Alpaca OCO rules mirror bracket leg logic).
        if long_side and not (tp > sl):
            tp, sl = sl + 0.02, tp - 0.02
        if (not long_side) and not (tp < sl):
            tp, sl = sl - 0.02, tp + 0.02

        log.info("%s: attaching OCO exits side=%s qty=%.4f | TP=%.2f SL=%.2f",
                 symbol, "long" if long_side else "short", qty, tp, sl)
        try:
            _submit_oco_exit(trading, symbol, long_side, qty, tp, sl)
        except APIError as e:
            log.warning("attach failed for %s: %s", symbol, e)

def main():
    trading = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    while True:
        try:
            run_once(trading, data_client)
        except Exception as e:
            log.exception("exit monitor loop error: %s", e)
        time.sleep(EXIT_MONITOR_POLL_SECONDS)

if __name__ == "__main__":
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        log.warning("ALPACA_API_KEY/SECRET not set; this script will fail to connect.")
    main()

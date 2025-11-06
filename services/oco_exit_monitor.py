import os
import time
import logging
import math
import uuid
from typing import Optional, Tuple, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus, OrderStatus
from alpaca.trading.requests import (
    LimitOrderRequest,
    StopOrderRequest,
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

# === ENV ===
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
EXIT_MONITOR_POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"

# Synthetic OCO for fractional qty
USE_SYNTHETIC_FRACTIONAL_OCO = os.getenv("USE_SYNTHETIC_FRACTIONAL_OCO", "1") == "1"
SYN_PREFIX = os.getenv("SYN_OCO_PREFIX", "fracoco")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def _qt(x: float, places: int = 2) -> float:
    return round(float(x) + 1e-9, places)

def _is_whole_share(qty: float) -> bool:
    return abs(qty - round(qty)) < 1e-9

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

def _derive_tp_sl(is_long: bool, ref_px: float, atr_val: Optional[float]) -> Tuple[float, float]:
    if atr_val is None:
        atr_val = 0.50
    if is_long:
        tp = _qt(ref_px + TP_ATR_MULT * atr_val, 2)            # higher
        sl = _qt(max(0.01, ref_px - SL_ATR_MULT * atr_val), 2) # lower
    else:
        tp = _qt(ref_px - TP_ATR_MULT * atr_val, 2)            # lower
        sl = _qt(ref_px + SL_ATR_MULT * atr_val, 2)            # higher
    return tp, sl

def _open_orders(trading: TradingClient, symbol: Optional[str] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol] if symbol else None)
    return trading.get_orders(filter=req)

def _has_any_exit_orders(trading: TradingClient, symbol: str) -> bool:
    """Consider exits present if there is ANY open SELL order for the symbol (limit or stop)."""
    try:
        for o in _open_orders(trading, symbol):
            if str(o.side).lower().endswith("sell"):
                return True
        return False
    except Exception:
        return False

def _submit_true_oco_long(trading: TradingClient, symbol: str, qty: float, tp: float, sl: float):
    """
    True OCO for long exits requires whole shares. Fractional will 422.
    """
    req = LimitOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        order_class="oco",
        extended_hours=False,
        # children:
        take_profit=None,  # set via kwargs below (SDK maps properly if passed as named args)
        stop_loss=None,
    )
    # The SDK requires passing child legs as kwargs on submit_order:
    if DRY_RUN:
        log.info("[DRY_RUN] true OCO SELL %s qty=%.4f tp=%.2f sl=%.2f", symbol, qty, tp, sl)
        return
    trading.submit_order(req, take_profit={"limit_price": _qt(tp, 2)}, stop_loss={"stop_price": _qt(sl, 2)})

def _submit_synthetic_oco_long(trading: TradingClient, symbol: str, qty: float, tp: float, sl: float):
    """
    Synthetic OCO for fractional: place two simple SELL orders:
      - limit SELL at TP
      - stop SELL at SL
    Monitor each loop: if one fills/partial-fills, cancel the sibling.
    """
    tag = f"{SYN_PREFIX}-{symbol}-{uuid.uuid4().hex[:6]}"
    # 1) TP limit
    req_tp = LimitOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        limit_price=_qt(tp, 2),
        extended_hours=False,
        client_order_id=f"{tag}-tp",
    )
    # 2) SL stop (simple stop – market on trigger)
    req_sl = StopOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        stop_price=_qt(sl, 2),
        extended_hours=False,
        client_order_id=f"{tag}-sl",
    )
    if DRY_RUN:
        log.info("[DRY_RUN] synthetic OCO SELL %s qty=%.4f | TP limit=%.2f | SL stop=%.2f", symbol, qty, tp, sl)
        return
    trading.submit_order(req_tp)
    trading.submit_order(req_sl)
    log.info("synthetic OCO submitted -> %s qty=%.4f TP=%.2f SL=%.2f tag=%s", symbol, qty, tp, sl, tag)

def _cancel_orders(trading: TradingClient, orders: List):
    for o in orders:
        try:
            trading.cancel_order_by_id(o.id)
        except Exception as e:
            log.warning("cancel %s failed: %s", o.client_order_id, e)

def _reconcile_synthetic_pairs(trading: TradingClient, symbol: str, position_qty: float):
    """
    If one of the synthetic legs filled, cancel its sibling.
    Also, if position_qty dropped to 0, cancel any remaining sell legs.
    """
    open_os = _open_orders(trading, symbol)
    syn_tp = [o for o in open_os if o.client_order_id and f"{SYN_PREFIX}-{symbol}" in o.client_order_id and o.client_order_id.endswith("-tp")]
    syn_sl = [o for o in open_os if o.client_order_id and f"{SYN_PREFIX}-{symbol}" in o.client_order_id and o.client_order_id.endswith("-sl")]

    # If position is flat, cancel all synthetic legs
    if position_qty <= 1e-9 and (syn_tp or syn_sl):
        log.info("%s: flat -> cancel remaining synthetic exits", symbol)
        _cancel_orders(trading, syn_tp + syn_sl)
        return

    # If either side is missing, nothing to do
    if not syn_tp or not syn_sl:
        return

    # Check if any tp/sl leg has moved out of OPEN state → cancel sibling
    for tp_o in syn_tp:
        if tp_o.status not in (OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED):
            # sibling must be canceled
            mates = [s for s in syn_sl if s.client_order_id.split("-tp")[0] == tp_o.client_order_id.split("-tp")[0]]
            _cancel_orders(trading, mates)
    for sl_o in syn_sl:
        if sl_o.status not in (OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED):
            mates = [t for t in syn_tp if t.client_order_id.split("-sl")[0] == sl_o.client_order_id.split("-sl")[0]]
            _cancel_orders(trading, mates)

def run_once(trading: TradingClient, data_client: StockHistoricalDataClient):
    positions = trading.get_all_positions()
    if not positions:
        log.info("no positions")
        return

    for p in positions:
        symbol = p.symbol
        is_long = float(p.qty) > 0
        qty = abs(float(p.qty))
        avg_entry = float(p.avg_entry_price)

        # Always reconcile synthetic legs first
        _reconcile_synthetic_pairs(trading, symbol, qty if is_long else -qty)

        # Skip if any SELL exits already present
        if _has_any_exit_orders(trading, symbol):
            log.info("%s: exits already present", symbol)
            continue

        bid, ask, mid = _latest_bid_ask_mid(data_client, symbol)
        ref_px = mid if mid is not None else avg_entry

        atr_val, close_ref = _atr(symbol)
        ref_for_tp_sl = ref_px if ref_px is not None else (close_ref if close_ref else avg_entry)
        tp, sl = _derive_tp_sl(is_long, ref_for_tp_sl, atr_val)

        # Sanity: ensure correct ordering for long/short
        if is_long and not (tp > sl):
            tp, sl = sl + 0.02, tp - 0.02
        if (not is_long) and not (tp < sl):
            tp, sl = sl - 0.02, tp + 0.02

        if is_long:
            # LONG exit with qty possibly fractional
            if _is_whole_share(qty):
                log.info("%s: attaching TRUE OCO (whole shares) qty=%.4f | TP=%.2f SL=%.2f", symbol, qty, tp, sl)
                try:
                    if not DRY_RUN:
                        _submit_true_oco_long(trading, symbol, qty, tp, sl)
                    else:
                        log.info("[DRY_RUN] would submit true OCO")
                except APIError as e:
                    log.warning("true OCO failed for %s: %s", symbol, e)
            else:
                if USE_SYNTHETIC_FRACTIONAL_OCO:
                    log.info("%s: attaching SYNTHETIC OCO (fractional) qty=%.4f | TP=%.2f SL=%.2f", symbol, qty, tp, sl)
                    try:
                        _submit_synthetic_oco_long(trading, symbol, qty, tp, sl)
                    except APIError as e:
                        log.warning("synthetic OCO failed for %s: %s", symbol, e)
                else:
                    log.info("%s: fractional position -> skipping OCO (set USE_SYNTHETIC_FRACTIONAL_OCO=1 to enable)", symbol)
        else:
            # SHORT exits (mirror if you short later): implement as needed
            log.info("%s: short exits not implemented yet", symbol)

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

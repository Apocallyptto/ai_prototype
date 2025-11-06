import os
import time
import logging
import math
import uuid
from typing import Optional, Tuple, List, Dict

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderType, QueryOrderStatus, OrderStatus
)
from alpaca.trading.requests import (
    LimitOrderRequest, StopOrderRequest, GetOrdersRequest
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

def _qqty(q: float) -> float:
    """Round quantity to 0.001 share precision."""
    return round(max(0.0, float(q)), 3)

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
                ref = float(close.iloc[-1])
                return float(atr) if not math.isnan(atr) else None, ref
        except Exception:
            pass
    return None, None

def _derive_tp_sl(is_long: bool, ref_px: float, atr_val: Optional[float]) -> Tuple[float, float]:
    if atr_val is None:
        atr_val = 0.50
    if is_long:
        tp = _qt(ref_px + TP_ATR_MULT * atr_val, 2)
        sl = _qt(max(0.01, ref_px - SL_ATR_MULT * atr_val), 2)
    else:
        tp = _qt(ref_px - TP_ATR_MULT * atr_val, 2)
        sl = _qt(ref_px + SL_ATR_MULT * atr_val, 2)
    return tp, sl

def _open_orders(trading: TradingClient, symbol: Optional[str] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol] if symbol else None)
    return trading.get_orders(filter=req)

def _classify_sell_orders(orders) -> Dict[str, float]:
    """
    Return:
      reserved_total: sum of ALL open SELL qty for the symbol
      syn_tp_qty: sum of synthetic TP qty (our tag)
      syn_sl_qty: sum of synthetic SL qty (our tag)
    """
    reserved_total = 0.0
    syn_tp_qty = 0.0
    syn_sl_qty = 0.0
    for o in orders:
        try:
            side = str(getattr(o, "side", "")).lower()
            if not side.endswith("sell"):
                continue
            q = float(getattr(o, "qty", 0) or 0.0)
            reserved_total += q
            coid = (getattr(o, "client_order_id", "") or "")
            if coid.startswith(f"{SYN_PREFIX}-") and coid.endswith("-tp"):
                syn_tp_qty += q
            if coid.startswith(f"{SYN_PREFIX}-") and coid.endswith("-sl"):
                syn_sl_qty += q
        except Exception:
            pass
    return {
        "reserved": reserved_total,
        "syn_tp_qty": syn_tp_qty,
        "syn_sl_qty": syn_sl_qty,
    }

def _submit_true_oco_long(trading: TradingClient, symbol: str, qty: float, tp: float, sl: float):
    req = LimitOrderRequest(
        symbol=symbol, side=OrderSide.SELL, type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY, qty=_qqty(qty), order_class="oco",
        extended_hours=False,
    )
    if DRY_RUN:
        log.info("[DRY_RUN] true OCO SELL %s qty=%.3f tp=%.2f sl=%.2f", symbol, qty, tp, sl)
        return
    trading.submit_order(req, take_profit={"limit_price": _qt(tp, 2)}, stop_loss={"stop_price": _qt(sl, 2)})

def _submit_simple_tp(trading: TradingClient, symbol: str, qty: float, tp: float, tag: str):
    req_tp = LimitOrderRequest(
        symbol=symbol, side=OrderSide.SELL, type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY, qty=_qqty(qty), limit_price=_qt(tp, 2),
        extended_hours=False, client_order_id=f"{tag}-tp",
    )
    if DRY_RUN:
        log.info("[DRY_RUN] TP simple SELL %s qty=%.3f @ %.2f (%s)", symbol, qty, tp, tag)
        return
    trading.submit_order(req_tp)

def _submit_simple_sl(trading: TradingClient, symbol: str, qty: float, sl: float, tag: str):
    req_sl = StopOrderRequest(
        symbol=symbol, side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY, qty=_qqty(qty), stop_price=_qt(sl, 2),
        extended_hours=False, client_order_id=f"{tag}-sl",
    )
    if DRY_RUN:
        log.info("[DRY_RUN] SL simple SELL %s qty=%.3f @ %.2f (%s)", symbol, qty, sl, tag)
        return
    trading.submit_order(req_sl)

def _cancel_orders(trading: TradingClient, orders: List):
    for o in orders:
        try:
            trading.cancel_order_by_id(o.id)
        except Exception as e:
            log.warning("cancel %s failed: %s", o.client_order_id, e)

def _reconcile_synthetic_pairs(trading: TradingClient, symbol: str, position_qty: float):
    open_os = _open_orders(trading, symbol)
    syn_tp = [o for o in open_os if o.client_order_id and f"{SYN_PREFIX}-{symbol}" in o.client_order_id and o.client_order_id.endswith("-tp")]
    syn_sl = [o for o in open_os if o.client_order_id and f"{SYN_PREFIX}-{symbol}" in o.client_order_id and o.client_order_id.endswith("-sl")]

    # If flat, cancel remaining sells
    if position_qty <= 1e-9 and (syn_tp or syn_sl):
        log.info("%s: flat -> cancel remaining synthetic exits", symbol)
        _cancel_orders(trading, syn_tp + syn_sl)
        return

    # If any TP/SL leg is no longer OPEN-ish, cancel its sibling(s)
    openish = {OrderStatus.NEW, OrderStatus.ACCEPTED, OrderStatus.PENDING_NEW, OrderStatus.PARTIALLY_FILLED}
    for tp_o in syn_tp:
        if tp_o.status not in openish:
            mates = [s for s in syn_sl if s.client_order_id.split("-tp")[0] == tp_o.client_order_id.split("-tp")[0]]
            _cancel_orders(trading, mates)
    for sl_o in syn_sl:
        if sl_o.status not in openish:
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
        pos_qty = abs(float(p.qty))
        avg_entry = float(p.avg_entry_price)

        # Reconcile synthetic legs first
        _reconcile_synthetic_pairs(trading, symbol, pos_qty if is_long else -pos_qty)

        # Only long exits implemented here
        if not is_long:
            log.info("%s: short exits not implemented yet", symbol)
            continue

        # Gather SELL orders to compute available headroom
        all_open = _open_orders(trading, symbol)
        sel = _classify_sell_orders(all_open)
        reserved = sel["reserved"]
        syn_tp_qty = sel["syn_tp_qty"]
        syn_sl_qty = sel["syn_sl_qty"]
        available = max(0.0, pos_qty - reserved)

        if syn_tp_qty > 0 or syn_sl_qty > 0:
            # At least one leg exists; try to add only the missing sibling with safe size
            need_tp = syn_tp_qty <= 1e-9
            need_sl = syn_sl_qty <= 1e-9
        else:
            # No legs exist yet; we may split available between both
            need_tp = True
            need_sl = True

        bid, ask, mid = _latest_bid_ask_mid(data_client, symbol)
        ref_px = mid if mid is not None else avg_entry
        atr_val, close_ref = _atr(symbol)
        ref_for_tp_sl = ref_px if ref_px is not None else (close_ref if close_ref else avg_entry)
        tp, sl = _derive_tp_sl(True, ref_for_tp_sl, atr_val)
        if not (tp > sl):
            tp, sl = sl + 0.02, tp - 0.02  # maintain proper ordering

        # Whole shares → can use native OCO if nothing exists yet and qty>=1
        if _is_whole_share(pos_qty) and need_tp and need_sl and available >= 1.0:
            log.info("%s: attaching TRUE OCO (whole shares) qty=%.3f | TP=%.2f SL=%.2f", symbol, pos_qty, tp, sl)
            try:
                if not DRY_RUN:
                    _submit_true_oco_long(trading, symbol, pos_qty, tp, sl)
            except APIError as e:
                log.warning("true OCO failed for %s: %s", symbol, e)
            continue

        # Fractional or partial availability → synthetic with sizing constraints
        if not USE_SYNTHETIC_FRACTIONAL_OCO:
            log.info("%s: fractional position but synthetic disabled -> skipping", symbol)
            continue

        tag = f"{SYN_PREFIX}-{symbol}-{uuid.uuid4().hex[:6]}"

        if need_tp and need_sl:
            # place BOTH, but sum must not exceed available; split 50/50
            if available <= 1e-6:
                log.info("%s: exits wanted but no available qty (reserved=%.3f pos=%.3f)", symbol, reserved, pos_qty)
                continue
            per_leg = _qqty(available / 2.0)
            if per_leg <= 0.0:
                # if rounding killed it, prefer to place just a STOP for safety
                sl_qty = _qqty(available)
                log.info("%s: placing only SL (safety) qty=%.3f @ %.2f", symbol, sl_qty, sl)
                try:
                    _submit_simple_sl(trading, symbol, sl_qty, sl, tag)
                except APIError as e:
                    log.warning("%s: SL submit failed: %s", symbol, e)
            else:
                log.info("%s: attaching SYN OCO both legs qty=%.3f each | TP=%.2f SL=%.2f (avail=%.3f)",
                         symbol, per_leg, tp, sl, available)
                try:
                    _submit_simple_tp(trading, symbol, per_leg, tp, tag)
                    _submit_simple_sl(trading, symbol, per_leg, sl, tag)
                except APIError as e:
                    log.warning("%s: synthetic both legs failed: %s", symbol, e)
        elif need_tp:
            # Only TP missing; size = min(existing SL qty, available) or available
            want = syn_sl_qty if syn_sl_qty > 0 else available
            qty = _qqty(min(want, available))
            if qty <= 0.0:
                log.info("%s: TP missing but no available qty (reserved=%.3f pos=%.3f)", symbol, reserved, pos_qty)
                return
            log.info("%s: attaching SYN TP qty=%.3f @ %.2f (avail=%.3f, existing SL=%.3f)",
                     symbol, qty, tp, available, syn_sl_qty)
            try:
                _submit_simple_tp(trading, symbol, qty, tp, tag)
            except APIError as e:
                log.warning("%s: TP submit failed: %s", symbol, e)
        elif need_sl:
            # Only SL missing; size = min(existing TP qty, available)
            want = syn_tp_qty if syn_tp_qty > 0 else available
            qty = _qqty(min(want, available))
            if qty <= 0.0:
                log.info("%s: SL missing but no available qty (reserved=%.3f pos=%.3f)", symbol, reserved, pos_qty)
                return
            log.info("%s: attaching SYN SL qty=%.3f @ %.2f (avail=%.3f, existing TP=%.3f)",
                     symbol, qty, sl, available, syn_tp_qty)
            try:
                _submit_simple_sl(trading, symbol, qty, sl, tag)
            except APIError as e:
                log.warning("%s: SL submit failed: %s", symbol, e)
        else:
            log.info("%s: exits already present", symbol)

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

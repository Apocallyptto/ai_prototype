import os
import time
import math
import logging
from typing import Optional, List

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    StopOrderRequest,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("oco_exit_monitor")

# --- ENV knobs ---
POLL_SECONDS = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
ALLOW_AFTER_HOURS = os.getenv("ALLOW_AFTER_HOURS", "1") == "1"  # for TP only; SL always extended_hours=False
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "1.00"))
MIN_QTY = float(os.getenv("MIN_QTY", "0.001"))
COOLDOWN_SECONDS = int(os.getenv("EXIT_COOLDOWN_SECONDS", "45"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

_last_action_ts = 0.0


# ---------- helpers ----------
def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)


def _qq(q: float) -> float:
    # 3 decimals for fractional stocks
    q = float(q)
    if q <= 0:
        return 0.0
    return math.floor(q * 1000.0 + 1e-9) / 1000.0


def _rth(tr: TradingClient) -> bool:
    try:
        clk = tr.get_clock()
        return bool(getattr(clk, "is_open", False))
    except Exception:
        return True  # fail-open so we don't block during outages


def _latest_mid(dc: StockHistoricalDataClient, symbol: str) -> Optional[float]:
    try:
        q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        bid = float(q.bid_price) if q.bid_price is not None else None
        ask = float(q.ask_price) if q.ask_price is not None else None
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2.0
    except Exception:
        pass
    return None


def _get_positions(tr: TradingClient):
    try:
        return tr.get_all_positions()
    except Exception:
        return []


def _open_orders(tr: TradingClient, symbol: str):
    return tr.get_orders(
        filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[symbol])
    )


def _min_notional_price(px: float, qty: float) -> float:
    if qty <= 0:
        return px
    need = MIN_NOTIONAL / qty
    return max(px, _qt(need, 2))


def _tp_sl_from_anchor(is_long: bool, anchor: float, atr: Optional[float]):
    a = atr if atr is not None else 0.50
    if is_long:
        tp = _qt(anchor + TP_ATR_MULT * a, 2)
        sl = _qt(max(0.01, anchor - SL_ATR_MULT * a), 2)
        if tp <= sl:
            tp = _qt(sl + 0.02, 2)
    else:
        tp = _qt(anchor - TP_ATR_MULT * a, 2)
        sl = _qt(anchor + SL_ATR_MULT * a, 2)
        if tp >= sl:
            sl = _qt(tp + 0.02, 2)
    return tp, sl


def _available_sell_qty(tr: TradingClient, symbol: str, pos_qty: float) -> float:
    """Reserve qty held by existing SELL orders; return free qty."""
    try:
        od = _open_orders(tr, symbol)
    except Exception:
        return pos_qty

    held = 0.0
    for o in od:
        try:
            if getattr(o, "side", None) == OrderSide.SELL:
                held += float(getattr(o, "qty", 0.0) or 0.0)
        except Exception:
            pass
    free = max(0.0, pos_qty - held)
    return _qq(free)


def _has_tp_sl(tr: TradingClient, symbol: str) -> (bool, bool):
    """Detect if we already have any TP/SL orders present for symbol."""
    tp_present = False
    sl_present = False
    try:
        for o in _open_orders(tr, symbol):
            t = getattr(o, "type", None)
            if t == OrderType.LIMIT and getattr(o, "side", None) == OrderSide.SELL:
                tp_present = True
            if t == OrderType.STOP and getattr(o, "side", None) == OrderSide.SELL:
                sl_present = True
    except Exception:
        pass
    return tp_present, sl_present


def _place_tp(tr: TradingClient, symbol: str, qty: float, limit_px: float, is_ah: bool):
    # TP is a LIMIT SELL; extended_hours allowed in AH
    qty = _qq(qty)
    if qty < MIN_QTY:
        return None
    limit_px = _qt(limit_px, 2)
    limit_px = _min_notional_price(limit_px, qty)
    req = LimitOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        qty=qty,
        time_in_force=TimeInForce.DAY,
        type=OrderType.LIMIT,
        limit_price=limit_px,
        extended_hours=(ALLOW_AFTER_HOURS and is_ah),
    )
    return tr.submit_order(req)


def _place_sl(tr: TradingClient, symbol: str, qty: float, stop_px: float):
    # SL is a STOP SELL; extended hours NOT allowed by Alpaca => always False
    qty = _qq(qty)
    if qty < MIN_QTY:
        return None
    stop_px = _qt(stop_px, 2)
    stop_px = _min_notional_price(stop_px, qty)  # no-op if already above $1 notional
    req = StopOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        qty=qty,
        time_in_force=TimeInForce.DAY,
        type=OrderType.STOP,
        stop_price=stop_px,
        extended_hours=False,  # <- critical fix
    )
    return tr.submit_order(req)


# ---------- main loop ----------
def run_once(tr: TradingClient, dc: StockHistoricalDataClient):
    global _last_action_ts
    now = time.time()
    if now - _last_action_ts < COOLDOWN_SECONDS:
        return

    is_rth = _rth(tr)
    poss = _get_positions(tr)
    if not poss:
        log.info("no positions")
        return

    for p in poss:
        try:
            symbol = p.symbol.upper()
            if SYMBOLS and symbol not in SYMBOLS:
                continue

            # we only manage long positions here
            qty = float(getattr(p, "qty", 0.0) or 0.0)
            if qty <= 0:
                continue

            # detect existing exit legs
            tp_present, sl_present = _has_tp_sl(tr, symbol)

            # if there is any OPEN entry (BUY) order, defer attaching exits
            ods = _open_orders(tr, symbol)
            has_open_buy = any(getattr(o, "side", None) == OrderSide.BUY for o in ods)
            if has_open_buy:
                # don't attach exits while entry is pending to avoid wash-trade conflicts
                open_ids = [o.id for o in ods if getattr(o, "side", None) == OrderSide.BUY]
                log.info(f"{symbol}: entry still open ({', '.join(open_ids)}); deferring exits")
                continue

            # compute free qty to protect (not already reserved by SELLs)
            free = _available_sell_qty(tr, symbol, qty)
            if free < MIN_QTY and tp_present and sl_present:
                log.info(f"{symbol}: exits already present")
                continue

            # pick an anchor (mid)
            mid = _latest_mid(dc, symbol) or float(getattr(p, "avg_entry_price", 0.0) or 0.0)
            if mid <= 0:
                log.info(f"{symbol}: no price; skipping")
                continue

            tp_px, sl_px = _tp_sl_from_anchor(is_long=True, anchor=mid, atr=None)

            if not tp_present and free >= MIN_QTY:
                # In AH we can at least place TP
                o = _place_tp(tr, symbol, free, tp_px, is_ah=(not is_rth))
                if o:
                    log.info(f"{symbol}: placed TP qty={_qq(free):.3f} @ {tp_px:.2f}")
                    _last_action_ts = time.time()

            # SL must be submitted without extended hours
            # If market is closed, request will still be accepted (it just won't trigger until RTH),
            # but we MUST NOT set extended_hours=True.
            # If we already consumed all free qty by TP, nothing left for SL — we split evenly if both missing.
            if not sl_present:
                # compute how much qty is still free after TP submit
                free_after_tp = _available_sell_qty(tr, symbol, qty)
                # If both legs missing, split qty roughly in half to avoid "held_for_orders" conflict
                if not tp_present:
                    free_after_tp = _qq(qty / 2.0)
                if free_after_tp >= MIN_QTY:
                    try:
                        o2 = _place_sl(tr, symbol, free_after_tp, sl_px)
                        if o2:
                            log.info(f"{symbol}: placed SL qty={_qq(free_after_tp):.3f} @ {sl_px:.2f}")
                            _last_action_ts = time.time()
                    except Exception as e:
                        # If we ever see the extended-hours error again, it's safe to just skip SL until RTH.
                        msg = str(getattr(e, "args", [""])[-1])
                        if "extended hours" in msg.lower():
                            log.info(f"{symbol}: market closed for STOP; will attach SL at RTH")
                        else:
                            log.warning(f"{symbol}: SL submit failed: {msg}")

            if tp_present and sl_present and free < MIN_QTY:
                log.info(f"{symbol}: exits already present")

        except Exception as e:
            log.exception(f"{symbol if 'symbol' in locals() else '?'}: monitor loop error: {e}")


def main():
    tr = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
    dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    while True:
        try:
            run_once(tr, dc)
        except Exception as e:
            log.exception(f"oco_exit_monitor fatal: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

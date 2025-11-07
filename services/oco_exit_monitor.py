import os
import time
import logging
from typing import List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("oco_exit_monitor")

POLL = int(os.getenv("EXIT_MONITOR_POLL_SECONDS", "15"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "1.5"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "1.00"))
COOLDOWN_SECONDS = int(os.getenv("EXIT_MONITOR_COOLDOWN_SECONDS", "30"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

# We use simple client_order_id suffixes to identify legs
SUF_TP = "-tp"
SUF_SL = "-sl"

_last_attempt = {}  # symbol -> epoch seconds


def _qt(x: float, p: int = 2) -> float:
    return round(float(x) + 1e-9, p)


def _cooldown_ok(symbol: str) -> bool:
    now = time.time()
    last = _last_attempt.get(symbol, 0)
    if now - last >= COOLDOWN_SECONDS:
        _last_attempt[symbol] = now
        return True
    return False


def _open_orders(tc: TradingClient, symbol: str):
    return tc.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
            symbols=[symbol],
        )
    )


def _existing_legs(tc: TradingClient, symbol: str):
    """Return (tp_orders, sl_orders) currently open on SELL side for this symbol."""
    od = _open_orders(tc, symbol)
    tps = []
    sls = []
    for o in od:
        try:
            coid = (o.client_order_id or "")
            if getattr(o, "side", None) == OrderSide.SELL:
                if coid.endswith(SUF_TP):
                    tps.append(o)
                elif coid.endswith(SUF_SL):
                    sls.append(o)
        except Exception:
            pass
    return tps, sls


def _has_open_entry(tc: TradingClient, symbol: str, long_side: bool) -> Optional[str]:
    """
    If there exists any OPEN order on the entry side (BUY for long, SELL for short),
    return its client_order_id (indicates we must defer exits). Otherwise None.
    """
    od = _open_orders(tc, symbol)
    wanted_side = OrderSide.BUY if long_side else OrderSide.SELL
    for o in od:
        try:
            if getattr(o, "side", None) == wanted_side:
                return o.client_order_id or o.id
        except Exception:
            pass
    return None


def _position_info(tc: TradingClient, symbol: str) -> Tuple[float, Optional[float]]:
    """
    Returns (qty, avg_entry_price or None)
    """
    for p in tc.get_all_positions():
        if p.symbol == symbol:
            try:
                return float(p.qty), float(p.avg_entry_price)
            except Exception:
                return float(p.qty), None
    return 0.0, None


def _qty_already_held_for_sell(tc: TradingClient, symbol: str) -> float:
    """Compute how much position qty is already reserved by SELL orders (both TP and SL)."""
    q = 0.0
    for o in _open_orders(tc, symbol):
        try:
            if getattr(o, "side", None) == OrderSide.SELL:
                q += float(o.qty)
        except Exception:
            pass
    return q


def _attach_synthetic_tp(tc: TradingClient, symbol: str, qty: float, limit_px: float):
    coid = f"fracoco-{symbol[:5]}-{os.urandom(3).hex()}{SUF_TP}"
    req = LimitOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        limit_price=_qt(limit_px, 2),
        extended_hours=True,   # allow during AH
        client_order_id=coid,
    )
    return tc.submit_order(req)


def _attach_synthetic_sl(tc: TradingClient, symbol: str, qty: float, stop_px: float):
    # Using STOP (market stop) as synthetic SL; fractional simple order is allowed.
    coid = f"fracoco-{symbol[:5]}-{os.urandom(3).hex()}{SUF_SL}"
    req = StopOrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        type=OrderType.STOP,
        time_in_force=TimeInForce.DAY,
        qty=qty,
        stop_price=_qt(stop_px, 2),
        extended_hours=True,   # allow during AH
        client_order_id=coid,
    )
    return tc.submit_order(req)


def _compute_tp_sl_from_ref(ref_px: float, atr: float = 0.5) -> Tuple[float, float]:
    tp = _qt(ref_px + TP_ATR_MULT * atr, 2)
    sl = _qt(max(0.01, ref_px - SL_ATR_MULT * atr), 2)
    if tp <= sl:
        tp, sl = sl + 0.02, max(0.01, tp - 0.02)
    return tp, sl


def run_once(tc: TradingClient, symbols: List[str]):
    for sym in symbols:
        qty, avg = _position_info(tc, sym)
        if qty <= 0:
            continue  # only manage long positions here

        # 1) If an entry BUY is still open, DEFER exits (prevents wash-trade 403)
        entry_open_id = _has_open_entry(tc, sym, long_side=True)
        if entry_open_id:
            log.info(f"{sym}: entry still open ({entry_open_id}); deferring exits")
            continue

        # 2) Determine how much is already reserved by existing SELLs
        held_sell_qty = _qty_already_held_for_sell(tc, sym)
        avail = max(0.0, _qt(qty - held_sell_qty, 3))
        tp_orders, sl_orders = _existing_legs(tc, sym)

        if avail <= 0 and tp_orders and sl_orders:
            log.info(f"{sym}: exits already present")
            continue

        # cooldown to avoid spam
        if not _cooldown_ok(sym):
            continue

        ref = avg if avg is not None else 0.0
        tp_px, sl_px = _compute_tp_sl_from_ref(ref, atr=0.5)

        # Ensure notional >= $1 where required
        if tp_px * max(avail, 0.0) < MIN_NOTIONAL:
            # if too small, try to place at least $1 notional TP/SL (round tiny top-up)
            min_qty = max(avail, _qt(MIN_NOTIONAL / max(tp_px, 0.01), 3))
            avail = min(avail, min_qty)

        # Attach missing legs only, sized to available qty
        if avail > 0:
            need_tp = len(tp_orders) == 0
            need_sl = len(sl_orders) == 0

            if need_tp and need_sl:
                # Place separately (synthetic OCO)
                log.info(f"{sym}: attaching SYN OCO both legs qty={_qt(avail/2,3)} each | TP={tp_px} SL={sl_px} (avail={_qt(avail,3)})")
                half = _qt(avail / 2.0, 3)
                # try both; if either fails with “insufficient qty”, we’ll retry next loop
                try:
                    _attach_synthetic_tp(tc, sym, half, tp_px)
                except Exception as e:
                    log.warning(f"{sym}: TP submit failed: {e}")
                try:
                    _attach_synthetic_sl(tc, sym, avail - half, sl_px)
                except Exception as e:
                    log.warning(f"{sym}: SL submit failed: {e}")

            elif need_tp:
                log.info(f"{sym}: attaching SYN TP qty={_qt(avail,3)} @ {tp_px} (avail path)")
                try:
                    _attach_synthetic_tp(tc, sym, avail, tp_px)
                except Exception as e:
                    log.warning(f"{sym}: TP submit failed: {e}")

            elif need_sl:
                log.info(f"{sym}: attaching SYN SL qty={_qt(avail,3)} @ {sl_px} (avail path)")
                try:
                    _attach_synthetic_sl(tc, sym, avail, sl_px)
                except Exception as e:
                    log.warning(f"{sym}: SL submit failed: {e}")
        else:
            log.info(f"{sym}: exits already present")


def main():
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

    # Manage only symbols we actually hold; you can restrict via env SYMBOLS if you want
    symbols_env = os.getenv("SYMBOLS")
    if symbols_env:
        symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    else:
        symbols = [p.symbol for p in tc.get_all_positions()]

    if not symbols:
        log.info("no positions")
        return

    while True:
        try:
            run_once(tc, symbols)
        except Exception as e:
            log.warning(f"monitor loop error: {e}")
        time.sleep(POLL)


if __name__ == "__main__":
    main()

# tools/ensure_exits_now.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide, OrderType, TimeInForce,
    QueryOrderStatus, OrderClass
)
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _qt_price(x: float) -> float:
    return round(float(x) + 1e-9, 2)


def _qt_qty(q: float) -> float:
    # allow fractional up to 0.001
    return max(0.0, int(float(q) * 1000 + 1e-9) / 1000.0)


@dataclass
class ExitCoverage:
    tp_qty: float = 0.0
    stop_qty: float = 0.0
    has_any_oco: bool = False


def _paper_flag() -> bool:
    tm = (os.getenv("TRADING_MODE", "paper") or "paper").lower().strip()
    ap = os.getenv("ALPACA_PAPER", "1")
    return (tm == "paper") or (ap != "0")


def _get_mid(dc: StockHistoricalDataClient, sym: str, fallback: float) -> float:
    q = dc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))[sym]
    bid = _f(q.bid_price, 0.0)
    ask = _f(q.ask_price, 0.0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return float(fallback)


def _scan_exit_coverage(open_orders_nested, sym: str, exit_side: OrderSide) -> ExitCoverage:
    cov = ExitCoverage()
    for o in open_orders_nested:
        if (o.symbol or "").upper() != sym:
            continue
        if o.side != exit_side:
            continue

        oq = _f(getattr(o, "qty", 0.0), 0.0)

        if getattr(o, "order_class", None) == OrderClass.OCO:
            cov.has_any_oco = True
            if getattr(o, "type", None) == OrderType.LIMIT:
                cov.tp_qty += oq

            legs = getattr(o, "legs", None) or []
            for l in legs:
                if getattr(l, "type", None) == OrderType.STOP:
                    lq = _f(getattr(l, "qty", None), oq)
                    cov.stop_qty += lq
        else:
            otype = getattr(o, "type", None)
            if otype == OrderType.LIMIT:
                cov.tp_qty += oq
            elif otype == OrderType.STOP:
                cov.stop_qty += oq

    return cov


def _cancel_symbol_side(tc: TradingClient, sym: str, side: OrderSide):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym])
    for o in tc.get_orders(req):
        if (o.symbol or "").upper() == sym and o.side == side:
            tc.cancel_order_by_id(o.id)


def _build_tp_sl(tp: float, sl: float):
    # Compatibility across alpaca-py versions
    try:
        from alpaca.trading.requests import TakeProfitRequest, StopLossRequest  # <-- correct in many versions
        return TakeProfitRequest(limit_price=tp), StopLossRequest(stop_price=sl)
    except Exception:
        return {"limit_price": tp}, {"stop_price": sl}


def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    sym = (argv[0] if argv else os.getenv("SYMBOL", "AAPL")).upper().strip()

    force_fix = (os.getenv("FORCE_FIX_EXITS", "0") == "1") or ("--force-fix" in argv)

    k = os.getenv("ALPACA_API_KEY")
    s = os.getenv("ALPACA_API_SECRET")
    if not k or not s:
        raise SystemExit("Missing ALPACA_API_KEY / ALPACA_API_SECRET")

    paper = _paper_flag()

    atr_pct = _f(os.getenv("ATR_PCT", "0.01"), 0.01)
    tp_mult = _f(os.getenv("TP_ATR_MULT", "1.0"), 1.0)
    sl_mult = _f(os.getenv("SL_ATR_MULT", "1.0"), 1.0)
    min_notional = _f(os.getenv("MIN_NOTIONAL", "1.00"), 1.00)
    allow_ah = (os.getenv("ALLOW_AFTER_HOURS", "0") == "1")

    tc = TradingClient(k, s, paper=paper)
    dc = StockHistoricalDataClient(k, s)

    pos = [p for p in tc.get_all_positions() if (p.symbol or "").upper() == sym]
    if not pos:
        print("no position")
        return

    qty = _f(pos[0].qty, 0.0)
    abs_qty = abs(qty)
    if abs_qty <= 0:
        print("no position")
        return

    is_long = qty > 0
    exit_side = OrderSide.SELL if is_long else OrderSide.BUY

    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True, symbols=[sym])
    open_orders = tc.get_orders(req)

    cov = _scan_exit_coverage(open_orders, sym, exit_side)

    if cov.tp_qty >= abs_qty and cov.stop_qty >= abs_qty:
        print("exits already present")
        return

    if cov.tp_qty >= abs_qty and cov.stop_qty < abs_qty:
        msg = f"[ensure_exits_now] WARNING: TP-only detected (tp_qty={cov.tp_qty}, stop_qty={cov.stop_qty}, abs_qty={abs_qty})"
        if not force_fix:
            print(msg + " → run with --force-fix (or FORCE_FIX_EXITS=1) to auto-repair.")
            return
        print(msg + " → auto-repair: canceling existing exit-side orders and placing OCO.")
        _cancel_symbol_side(tc, sym, exit_side)
        open_orders = tc.get_orders(req)
        cov = _scan_exit_coverage(open_orders, sym, exit_side)

    covered = min(cov.tp_qty, cov.stop_qty) if (cov.tp_qty > 0 and cov.stop_qty > 0) else 0.0
    remaining = _qt_qty(abs_qty - covered)
    if remaining <= 0:
        print("exits already present")
        return

    avg_entry = _f(getattr(pos[0], "avg_entry_price", 0.0), 0.0)
    mid = _get_mid(dc, sym, fallback=avg_entry if avg_entry > 0 else 1.0)

    if is_long:
        tp = _qt_price(mid * (1.0 + atr_pct * tp_mult))
        sl = _qt_price(max(0.01, mid * (1.0 - atr_pct * sl_mult)))
    else:
        tp = _qt_price(max(0.01, mid * (1.0 - atr_pct * tp_mult)))
        sl = _qt_price(mid * (1.0 + atr_pct * sl_mult))

    if tp * remaining < min_notional:
        tp = _qt_price(min_notional / remaining)
    if sl * remaining < min_notional:
        sl = _qt_price(min_notional / remaining)

    print(f"[ensure_exits_now] {sym} pos={'LONG' if is_long else 'SHORT'} qty={qty} abs_qty={abs_qty} held_tp={cov.tp_qty} held_sl={cov.stop_qty} free={remaining}")
    print(f"[ensure_exits_now] mid={mid:.4f} atr_pct={atr_pct:.4f} tp={tp} sl={sl} exit_side={exit_side.value} tif=day paper={paper} allow_ah={allow_ah}")

    tp_req, sl_req = _build_tp_sl(tp, sl)

    oco = tc.submit_order(
        LimitOrderRequest(
            symbol=sym,
            side=exit_side,
            qty=remaining,
            time_in_force=TimeInForce.DAY,
            limit_price=tp,
            order_class=OrderClass.OCO,
            take_profit=tp_req,
            stop_loss=sl_req,
            extended_hours=allow_ah,
        )
    )

    print("OCO submitted:", oco.id, "status:", oco.status, "order_class:", oco.order_class)


if __name__ == "__main__":
    main()

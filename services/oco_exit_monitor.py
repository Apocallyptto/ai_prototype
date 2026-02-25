import os
import time
import logging
from typing import Optional, List, Set, Dict

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    QueryOrderStatus,
    PositionIntent,
    OrderType,
)
from alpaca.trading.requests import (
    GetOrdersRequest,
    StopOrderRequest,
    MarketOrderRequest,
)

# Optional (native OCO for non-fractional qty). Safe to skip if SDK lacks these symbols.
try:
    from alpaca.trading.enums import OrderClass
    from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
except Exception:  # pragma: no cover
    OrderClass = None
    LimitOrderRequest = None
    TakeProfitRequest = None
    StopLossRequest = None

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOG = logging.getLogger("oco_exit_monitor")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return float(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def _resolve_mode() -> str:
    return (os.getenv("TRADING_MODE") or "paper").strip().lower()


def _resolve_paper() -> bool:
    if os.getenv("ALPACA_PAPER") is not None:
        return _env_bool("ALPACA_PAPER", default=True)
    return _resolve_mode() != "live"


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return TradingClient(key, secret, paper=_resolve_paper())


def get_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return StockHistoricalDataClient(key, secret)


def _round2(x: float) -> float:
    return float(f"{float(x):.2f}")


def _atr_pct(dc: StockHistoricalDataClient, sym: str, lookback: int = 50) -> Optional[float]:
    """Return ATR/close (e.g. 0.0012 = 0.12%)."""
    try:
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, limit=lookback)
        bars = dc.get_stock_bars(req).data.get(sym, [])
        if not bars or len(bars) < 2:
            return None

        trs = []
        prev_close = float(bars[0].close)
        for b in bars[1:]:
            high = float(b.high)
            low = float(b.low)
            close = float(b.close)
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
            prev_close = close

        if not trs:
            return None

        atr = sum(trs) / float(len(trs))
        last_close = float(bars[-1].close)
        if last_close <= 0:
            return None
        return atr / last_close
    except Exception as e:
        LOG.warning("atr_pct error %s: %s", sym, e)
        return None


def _get_open_orders(tc: TradingClient, symbols: Optional[List[str]] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True, symbols=symbols)
    return tc.get_orders(req) or []


def _cid(o) -> str:
    return (getattr(o, "client_order_id", "") or "")


def _pos_info(tc: TradingClient, sym: str) -> tuple[float, float, float]:
    """(qty, avg_entry, current_price) for symbol; 0s if not found."""
    symu = sym.upper()
    for p in (tc.get_all_positions() or []):
        if (getattr(p, "symbol", "") or "").upper() == symu:
            qty = float(getattr(p, "qty", 0) or 0)
            avg = float(getattr(p, "avg_entry_price", 0) or 0)
            cur = float(getattr(p, "current_price", 0) or 0)
            return qty, avg, cur
    return 0.0, 0.0, 0.0


def _cancel_order_safely(tc: TradingClient, oid: str, sym: str, why: str) -> None:
    try:
        tc.cancel_order_by_id(oid)
        LOG.info("canceled_order | sym=%s oid=%s why=%s", sym, oid, why)
    except Exception as e:
        LOG.warning("cancel_failed | sym=%s oid=%s why=%s err=%s", sym, oid, why, e)


def _cleanup_orphan_exit_orders(tc: TradingClient, open_orders, active_symbols: Set[str], prefix: str) -> int:
    canceled = 0
    for o in open_orders:
        sym = (getattr(o, "symbol", "") or "").upper()
        cid = _cid(o)
        if not cid.startswith(prefix):
            continue
        if sym not in active_symbols:
            oid = str(getattr(o, "id", "") or "")
            if oid:
                _cancel_order_safely(tc, oid, sym, "orphan_exit_no_position")
                canceled += 1
    return canceled


def _ensure_tp_sl(exit_side: OrderSide, tp: float, sl: float) -> tuple[float, float]:
    """
    exit_side SELL: closing long -> want TP > SL
    exit_side BUY : closing short -> want TP < SL
    """
    tp = float(tp)
    sl = float(sl)
    if exit_side == OrderSide.SELL:
        if tp <= sl:
            tp, sl = max(tp, sl), min(tp, sl)
    else:
        if tp >= sl:
            tp, sl = min(tp, sl), max(tp, sl)
    return tp, sl


def _tp_hit(exit_side: OrderSide, current_price: float, tp: float) -> bool:
    if current_price <= 0:
        return False
    if exit_side == OrderSide.SELL:
        return current_price >= tp
    return current_price <= tp


def _sl_cid_prefix(prefix: str, sym: str) -> str:
    return f"{prefix}{sym}-SL-"


def _find_exit_sl_orders(open_orders, sym: str, prefix: str) -> list:
    symu = sym.upper()
    pref = _sl_cid_prefix(prefix, symu)
    out = []
    for o in open_orders:
        if (getattr(o, "symbol", "") or "").upper() != symu:
            continue
        if not _cid(o).startswith(pref):
            continue
        out.append(o)
    return out


def _place_sl_simple(
    tc: TradingClient,
    sym: str,
    qty_abs: float,
    exit_side: OrderSide,
    intent: PositionIntent,
    sl_price: float,
    tif: TimeInForce,
    cid: str,
    dry_run: bool,
) -> bool:
    sl_price = _round2(sl_price)
    if dry_run:
        LOG.info("DRY_RUN place_sl | sym=%s qty=%s side=%s intent=%s sl=%s tif=%s cid=%s",
                 sym, qty_abs, exit_side.value, intent.value, sl_price, tif.value, cid)
        return True
    try:
        req = StopOrderRequest(
            symbol=sym,
            qty=float(qty_abs),
            side=exit_side,
            time_in_force=tif,
            stop_price=str(sl_price),
            position_intent=intent,
            client_order_id=cid,
        )
        o = tc.submit_order(req)
        oid = str(getattr(o, "id", "") or "")
        LOG.info("placed_sl | sym=%s qty=%s side=%s intent=%s sl=%s oid=%s cid=%s",
                 sym, qty_abs, exit_side.value, intent.value, sl_price, oid, cid)
        return True
    except Exception as e:
        LOG.error("place_sl failed sym=%s: %s", sym, e, exc_info=True)
        return False


def _close_market(
    tc: TradingClient,
    sym: str,
    qty_abs: float,
    exit_side: OrderSide,
    intent: PositionIntent,
    tif: TimeInForce,
    cid: str,
    dry_run: bool,
) -> bool:
    if dry_run:
        LOG.info("DRY_RUN close_market | sym=%s qty=%s side=%s intent=%s tif=%s cid=%s",
                 sym, qty_abs, exit_side.value, intent.value, tif.value, cid)
        return True
    try:
        req = MarketOrderRequest(
            symbol=sym,
            qty=float(qty_abs),
            side=exit_side,
            time_in_force=tif,
            type=OrderType.MARKET,
            position_intent=intent,
            client_order_id=cid,
        )
        o = tc.submit_order(req)
        oid = str(getattr(o, "id", "") or "")
        LOG.info("submitted_close_market | sym=%s qty=%s side=%s intent=%s oid=%s cid=%s",
                 sym, qty_abs, exit_side.value, intent.value, oid, cid)
        return True
    except Exception as e:
        LOG.error("close_market failed sym=%s: %s", sym, e, exc_info=True)
        return False


def _place_exit_oco_non_fractional(
    tc: TradingClient,
    sym: str,
    qty_abs: float,
    exit_side: OrderSide,
    intent: PositionIntent,
    tp_price: float,
    sl_price: float,
    tif: TimeInForce,
    allow_ah: bool,
    cid: str,
    dry_run: bool,
) -> bool:
    """Native OCO (non-fractional only). Fractional qty cannot use OCO: 422."""
    if OrderClass is None or LimitOrderRequest is None or TakeProfitRequest is None or StopLossRequest is None:
        LOG.warning("native_oco_not_available_in_sdk | sym=%s", sym)
        return False

    tp_price, sl_price = _ensure_tp_sl(exit_side, tp_price, sl_price)
    tp_price = _round2(tp_price)
    sl_price = _round2(sl_price)

    if dry_run:
        LOG.info(
            "DRY_RUN place_exit_oco | sym=%s qty=%s side=%s intent=%s tp=%s sl=%s tif=%s allow_ah=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, tp_price, sl_price, tif.value, allow_ah, cid
        )
        return True

    try:
        req = LimitOrderRequest(
            symbol=sym,
            qty=float(qty_abs),
            side=exit_side,
            time_in_force=tif,
            limit_price=tp_price,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
            position_intent=intent,
            client_order_id=cid,
            extended_hours=bool(allow_ah) and tif == TimeInForce.DAY,
        )
        o = tc.submit_order(req)
        oid = str(getattr(o, "id", "") or "")
        LOG.info(
            "placed_exit_oco | sym=%s qty=%s side=%s intent=%s tp=%s sl=%s oid=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, tp_price, sl_price, oid, cid
        )
        return True
    except Exception as e:
        LOG.error("place_exit_oco failed sym=%s: %s", sym, e, exc_info=True)
        return False


def main():
    poll_seconds = _env_int("POLL_SECONDS", 5)
    heartbeat_seconds = _env_int("HEARTBEAT_SECONDS", 60)
    error_backoff_seconds = _env_int("ERROR_BACKOFF_SECONDS", 30)

    min_qty = float(os.getenv("MIN_QTY", "0") or "0")
    qty_buffer_pct = _env_float("QTY_BUFFER_PCT", 0.0)

    use_atr = _env_bool("USE_ATR", True)
    atr_lookback = _env_int("ATR_LOOKBACK", 50)
    atr_mult_tp = _env_float("ATR_MULT_TP", 2.0)
    atr_mult_sl = _env_float("ATR_MULT_SL", 1.5)

    tp_pct = _env_float("TP_PCT", 1.0)
    sl_pct = _env_float("SL_PCT", 0.75)

    allow_ah = _env_bool("ALLOW_AH", False)

    tif_raw = (os.getenv("TIF", "day") or "day").strip().lower()
    try:
        tif = TimeInForce(tif_raw)
    except Exception:
        tif = TimeInForce.DAY

    enable_repair = _env_bool("ENABLE_REPAIR", True)
    dry_run = _env_bool("DRY_RUN", False)
    cleanup_orphans = _env_bool("CLEANUP_ORPHAN_EXITS", True)

    prefix = (os.getenv("EXIT_PREFIX", "EXIT-OCO-") or "EXIT-OCO-").strip()

    mode = _resolve_mode()
    paper = _resolve_paper()
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    LOG.info(
        "oco_exit_monitor starting | mode=%s | paper=%s | base=%s | poll=%ss | heartbeat=%ss | min_qty=%s | qty_buffer_pct=%s | "
        "use_atr=%s | atr_lookback=%s | atr_mult_tp=%s | atr_mult_sl=%s | tp_pct=%s | sl_pct=%s | allow_ah=%s | tif=%s | "
        "enable_repair=%s | dry_run=%s | cleanup_orphans=%s | prefix=%s | error_backoff=%ss",
        mode, paper, base, poll_seconds, heartbeat_seconds, min_qty, qty_buffer_pct,
        use_atr, atr_lookback, atr_mult_tp, atr_mult_sl, tp_pct, sl_pct, allow_ah, tif.value,
        enable_repair, dry_run, cleanup_orphans, prefix, error_backoff_seconds
    )

    tc = get_trading_client()
    dc = get_data_client()

    last_heartbeat = 0.0
    errors = 0
    last_fail_ts: Dict[str, float] = {}

    while True:
        try:
            positions = tc.get_all_positions() or []
            open_orders = _get_open_orders(tc)

            protected = 0
            placed = 0
            unprotected = []

            active_syms: Set[str] = set()
            for p in positions:
                sym = (getattr(p, "symbol", "") or "").upper()
                qty = float(getattr(p, "qty", 0) or 0)
                if qty != 0:
                    active_syms.add(sym)

            if cleanup_orphans:
                _cleanup_orphan_exit_orders(tc, open_orders, active_syms, prefix)
                open_orders = _get_open_orders(tc)

            for sym in sorted(active_syms):
                now = time.time()
                if sym in last_fail_ts and (now - last_fail_ts[sym]) < error_backoff_seconds:
                    continue

                qty, avg_entry, cur_price = _pos_info(tc, sym)
                if qty == 0:
                    continue

                qty_abs = abs(qty)
                if qty_abs < min_qty:
                    continue

                qty_eff = qty_abs * (1.0 - qty_buffer_pct / 100.0)
                if qty_eff <= 0:
                    continue

                pos_is_long = qty > 0
                exit_side = OrderSide.SELL if pos_is_long else OrderSide.BUY
                intent = PositionIntent.SELL_TO_CLOSE if pos_is_long else PositionIntent.BUY_TO_CLOSE

                ref_price = avg_entry if avg_entry > 0 else cur_price
                if ref_price <= 0 or cur_price <= 0:
                    continue

                # compute tp/sl
                if use_atr:
                    atrp = _atr_pct(dc, sym, lookback=atr_lookback)
                    if atrp is None:
                        if pos_is_long:
                            tp = ref_price * (1.0 + tp_pct / 100.0)
                            sl = ref_price * (1.0 - sl_pct / 100.0)
                        else:
                            tp = ref_price * (1.0 - tp_pct / 100.0)
                            sl = ref_price * (1.0 + sl_pct / 100.0)
                    else:
                        if pos_is_long:
                            tp = ref_price * (1.0 + atrp * atr_mult_tp)
                            sl = ref_price * (1.0 - atrp * atr_mult_sl)
                        else:
                            tp = ref_price * (1.0 - atrp * atr_mult_tp)
                            sl = ref_price * (1.0 + atrp * atr_mult_sl)
                else:
                    if pos_is_long:
                        tp = ref_price * (1.0 + tp_pct / 100.0)
                        sl = ref_price * (1.0 - sl_pct / 100.0)
                    else:
                        tp = ref_price * (1.0 - tp_pct / 100.0)
                        sl = ref_price * (1.0 + sl_pct / 100.0)

                tp, sl = _ensure_tp_sl(exit_side, tp, sl)

                # Key rule: your current trades are fractional -> only simple orders.
                is_fractional = qty_abs < 1.0

                if is_fractional:
                    # If TP hit: cancel SL then market-close (simple orders only).
                    if _tp_hit(exit_side, cur_price, tp):
                        sl_orders = _find_exit_sl_orders(open_orders, sym, prefix)
                        for o in sl_orders:
                            oid = str(getattr(o, "id", "") or "")
                            if oid:
                                _cancel_order_safely(tc, oid, sym, "tp_trigger_cancel_sl")
                        time.sleep(0.4)

                        qty2, _, _ = _pos_info(tc, sym)
                        if abs(qty2) < 1e-9:
                            protected += 1
                            continue

                        ok = _close_market(
                            tc=tc,
                            sym=sym,
                            qty_abs=abs(qty2),
                            exit_side=exit_side,
                            intent=intent,
                            tif=tif,
                            cid=f"{prefix}{sym}-TPMKT-{int(time.time())}",
                            dry_run=dry_run,
                        )
                        if ok:
                            placed += 1
                        else:
                            errors += 1
                            last_fail_ts[sym] = time.time()
                        continue

                    # Otherwise ensure protective SL exists.
                    sl_orders = _find_exit_sl_orders(open_orders, sym, prefix)
                    if sl_orders:
                        protected += 1
                        continue

                    if not enable_repair:
                        unprotected.append(sym)
                        continue

                    ok = _place_sl_simple(
                        tc=tc,
                        sym=sym,
                        qty_abs=qty_eff,
                        exit_side=exit_side,
                        intent=intent,
                        sl_price=sl,
                        tif=tif,
                        cid=f"{prefix}{sym}-SL-{int(time.time())}",
                        dry_run=dry_run,
                    )
                    if ok:
                        placed += 1
                    else:
                        errors += 1
                        last_fail_ts[sym] = time.time()
                    continue

                # Non-fractional: native OCO (rare in your setup)
                # If you ever trade whole shares, this path will protect with OCO.
                has_exit = False
                oco_pref = f"{prefix}{sym}-OCO-"
                for o in open_orders:
                    if (getattr(o, "symbol", "") or "").upper() != sym:
                        continue
                    if _cid(o).startswith(oco_pref):
                        has_exit = True
                        break
                    if str(getattr(o, "order_class", "") or "").lower() == "oco":
                        has_exit = True
                        break

                if has_exit:
                    protected += 1
                    continue

                if not enable_repair:
                    unprotected.append(sym)
                    continue

                ok = _place_exit_oco_non_fractional(
                    tc=tc,
                    sym=sym,
                    qty_abs=qty_eff,
                    exit_side=exit_side,
                    intent=intent,
                    tp_price=tp,
                    sl_price=sl,
                    tif=tif,
                    allow_ah=allow_ah,
                    cid=f"{prefix}{sym}-OCO-{int(time.time())}",
                    dry_run=dry_run,
                )
                if ok:
                    placed += 1
                else:
                    errors += 1
                    last_fail_ts[sym] = time.time()

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                open_orders = _get_open_orders(tc)
                LOG.info(
                    "Heartbeat | positions=%s protected=%s placed=%s open_orders=%s errors=%s | unprotected=%s",
                    len(positions), protected, placed, len(open_orders), errors, unprotected
                )

        except Exception:
            errors += 1
            LOG.error("cycle_error", exc_info=True)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
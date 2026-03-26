import os
import time
import logging
from typing import Optional, List, Set, Dict, Tuple

from datetime import datetime, timezone, timedelta

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

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sqlalchemy import text

from tools.db import get_engine

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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt_utc(dt) -> Optional[datetime]:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return None
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_side_text(v) -> str:
    s = str(v or "").strip().lower()
    if s.endswith(".buy") or s == "buy":
        return "buy"
    if s.endswith(".sell") or s == "sell":
        return "sell"
    return s


def _normalize_status_text(v) -> str:
    s = str(v or "").strip().lower()
    if s.endswith(".filled") or s == "filled":
        return "filled"
    if s.endswith(".canceled") or s == "canceled":
        return "canceled"
    if s.endswith(".cancelled") or s == "cancelled":
        return "canceled"
    if s.endswith(".new") or s == "new":
        return "new"
    if s.endswith(".open") or s == "open":
        return "open"
    return s


def _should_log_throttled(bucket: Dict[str, float], key: str, min_interval_sec: float = 60.0) -> bool:
    now = time.time()
    prev = bucket.get(key, 0.0)
    if now - prev >= min_interval_sec:
        bucket[key] = now
        return True
    return False


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


def _get_orders(tc: TradingClient, status: QueryOrderStatus, symbols: Optional[List[str]] = None):
    req = GetOrdersRequest(status=status, limit=500, nested=True, symbols=symbols)
    return tc.get_orders(filter=req) or []


def _cid(o) -> str:
    return (getattr(o, "client_order_id", "") or "")


def _pos_info(tc: TradingClient, sym: str) -> tuple[float, float, float]:
    """(qty, avg_entry, current_price)."""
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
        msg = str(e).lower()
        if 'already in "filled" state' in msg or "already in 'filled' state" in msg or 'already in "canceled" state' in msg:
            LOG.info("cancel_already_terminal | sym=%s oid=%s why=%s err=%s", sym, oid, why, e)
        else:
            LOG.warning("cancel_failed | sym=%s oid=%s why=%s err=%s", sym, oid, why, e)


def _cleanup_orphan_exit_orders(tc: TradingClient, orders, active_symbols: Set[str], prefix: str) -> int:
    canceled = 0
    for o in orders:
        sym = (getattr(o, "symbol", "") or "").upper()
        if not _cid(o).startswith(prefix):
            continue
        if sym not in active_symbols:
            oid = str(getattr(o, "id", "") or "")
            if oid:
                _cancel_order_safely(tc, oid, sym, "orphan_exit_no_position")
                canceled += 1
    return canceled


def _ensure_tp_sl(exit_side: OrderSide, tp: float, sl: float) -> tuple[float, float]:
    tp = float(tp)
    sl = float(sl)
    if exit_side == OrderSide.SELL:
        if tp <= sl:
            tp, sl = max(tp, sl), min(tp, sl)
    else:
        if tp >= sl:
            tp, sl = min(tp, sl), max(tp, sl)
    return tp, sl


def _clamp_sl_away_from_market(exit_side: OrderSide, sl: float, cur: float, min_gap_pct: float) -> float:
    if cur <= 0:
        return sl
    gap = max(0.0, float(min_gap_pct)) / 100.0
    if exit_side == OrderSide.SELL:
        max_sl = cur * (1.0 - gap)
        if sl >= max_sl:
            sl = max_sl
    else:
        min_sl = cur * (1.0 + gap)
        if sl <= min_sl:
            sl = min_sl
    return sl


def _tp_hit(exit_side: OrderSide, cur: float, tp: float) -> bool:
    if cur <= 0:
        return False
    if exit_side == OrderSide.SELL:
        return cur >= tp
    return cur <= tp


def _find_sl_orders(open_orders, sym: str, prefix: str) -> list:
    symu = sym.upper()
    pref = f"{prefix}{symu}-SL-"
    return [o for o in open_orders if (getattr(o, "symbol", "") or "").upper() == symu and _cid(o).startswith(pref)]


def _place_sl(
    tc: TradingClient,
    sym: str,
    qty_abs: float,
    exit_side: OrderSide,
    intent: PositionIntent,
    sl: float,
    tif: TimeInForce,
    cid: str,
    dry_run: bool,
) -> bool:
    sl = _round2(sl)
    if dry_run:
        LOG.info(
            "DRY_RUN place_sl | sym=%s qty=%s side=%s intent=%s sl=%s tif=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, sl, tif.value, cid
        )
        return True
    try:
        req = StopOrderRequest(
            symbol=sym,
            qty=float(qty_abs),
            side=exit_side,
            time_in_force=tif,
            stop_price=str(sl),
            position_intent=intent,
            client_order_id=cid,
        )
        o = tc.submit_order(order_data=req)
        oid = str(getattr(o, "id", "") or "")
        LOG.info(
            "placed_sl | sym=%s qty=%s side=%s intent=%s sl=%s tif=%s oid=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, sl, tif.value, oid, cid
        )
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
    cid: str,
    dry_run: bool,
) -> bool:
    if dry_run:
        LOG.info(
            "DRY_RUN close_market | sym=%s qty=%s side=%s intent=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, cid
        )
        return True
    try:
        req = MarketOrderRequest(
            symbol=sym,
            qty=float(qty_abs),
            side=exit_side,
            time_in_force=TimeInForce.DAY,
            type=OrderType.MARKET,
            position_intent=intent,
            client_order_id=cid,
        )
        o = tc.submit_order(order_data=req)
        oid = str(getattr(o, "id", "") or "")
        LOG.info(
            "submitted_close_market | sym=%s qty=%s side=%s intent=%s oid=%s cid=%s",
            sym, qty_abs, exit_side.value, intent.value, oid, cid
        )
        return True
    except Exception as e:
        LOG.error("close_market failed sym=%s: %s", sym, e, exc_info=True)
        return False


def _last_filled_entry_ts_from_db(db_engine, sym: str, pos_is_long: bool, lookback_days: int, prefix: str) -> Optional[datetime]:
    if db_engine is None:
        return None

    symu = sym.upper()
    side_like = "%buy" if pos_is_long else "%sell"

    try:
        with db_engine.connect() as con:
            row = con.execute(
                text(
                    """
                    SELECT recorded_at
                    FROM alpaca_orders
                    WHERE UPPER(symbol) = :sym
                      AND recorded_at >= (NOW() AT TIME ZONE 'UTC') - (:lookback_days || ' days')::interval
                      AND LOWER(CAST(status AS TEXT)) LIKE '%filled'
                      AND LOWER(CAST(side AS TEXT)) LIKE :side_like
                      AND COALESCE(client_order_id, '') NOT ILIKE :exit_prefix_like
                    ORDER BY recorded_at DESC
                    LIMIT 1
                    """
                ),
                {
                    "sym": symu,
                    "lookback_days": int(lookback_days),
                    "side_like": side_like,
                    "exit_prefix_like": f"{prefix}%",
                },
            ).fetchone()

        if not row:
            return None
        return _dt_utc(row[0])
    except Exception as e:
        LOG.warning("entry_ts lookup failed (db) | sym=%s err=%r", symu, e)
        return None


def _last_filled_entry_ts_from_api(tc: TradingClient, sym: str, pos_is_long: bool, lookback_days: int = 14) -> Optional[datetime]:
    """
    Find latest filled entry order timestamp:
    - long positions: last FILLED BUY for symbol
    - short positions: last FILLED SELL for symbol
    """
    symu = sym.upper()
    wanted_side = "buy" if pos_is_long else "sell"
    after = _utc_now() - timedelta(days=lookback_days)

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500, nested=True, after=after)
        orders = tc.get_orders(filter=req) or []
    except Exception as e:
        LOG.warning("entry_ts lookup failed (orders fetch) | sym=%s err=%r", symu, e)
        return None

    best: Optional[datetime] = None
    for o in orders:
        if (getattr(o, "symbol", "") or "").upper() != symu:
            continue

        side = _normalize_side_text(getattr(o, "side", ""))
        status = _normalize_status_text(getattr(o, "status", ""))

        if side != wanted_side:
            continue
        if status != "filled":
            continue

        ts = (
            _dt_utc(getattr(o, "filled_at", None))
            or _dt_utc(getattr(o, "submitted_at", None))
            or _dt_utc(getattr(o, "updated_at", None))
        )
        if not ts:
            continue

        if best is None or ts > best:
            best = ts

    return best


def _recover_entry_ts(
    tc: TradingClient,
    db_engine,
    sym: str,
    pos_is_long: bool,
    lookback_days: int,
    prefix: str,
) -> Optional[datetime]:
    ts = _last_filled_entry_ts_from_db(
        db_engine=db_engine,
        sym=sym,
        pos_is_long=pos_is_long,
        lookback_days=lookback_days,
        prefix=prefix,
    )
    if ts:
        return ts

    return _last_filled_entry_ts_from_api(
        tc=tc,
        sym=sym,
        pos_is_long=pos_is_long,
        lookback_days=lookback_days,
    )


def main():
    poll_seconds = _env_int("POLL_SECONDS", 5)
    heartbeat_seconds = _env_int("HEARTBEAT_SECONDS", 60)
    error_backoff_seconds = _env_int("ERROR_BACKOFF_SECONDS", 30)

    use_atr = _env_bool("USE_ATR", True)
    atr_lookback = _env_int("ATR_LOOKBACK", 50)
    atr_mult_tp = _env_float("ATR_MULT_TP", 2.0)
    atr_mult_sl = _env_float("ATR_MULT_SL", 1.5)

    tp_pct = _env_float("TP_PCT", 5.0)
    sl_pct = _env_float("SL_PCT", 3.0)

    min_sl_gap_pct = _env_float("MIN_SL_GAP_PCT", 0.20)

    enable_repair = _env_bool("ENABLE_REPAIR", True)
    dry_run = _env_bool("DRY_RUN", False)
    cleanup_orphans = _env_bool("CLEANUP_ORPHAN_EXITS", True)

    # fractional behavior
    fractional_tp_market_close = _env_bool("FRACTIONAL_TP_MARKET_CLOSE", True)
    fractional_threshold = float(os.getenv("FRACTIONAL_QTY_THRESHOLD", "1.0"))

    # TP min-hold (prevents daytrades on quick TP)
    tp_min_hold_minutes = int(os.getenv("TP_MIN_HOLD_MINUTES", os.getenv("MIN_HOLD_MINUTES", "0")))
    entry_lookup_days = _env_int("ENTRY_LOOKBACK_DAYS", 14)

    prefix = (os.getenv("EXIT_PREFIX", "EXIT-OCO-") or "EXIT-OCO-").strip()

    mode = _resolve_mode()
    paper = _resolve_paper()
    base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    LOG.info(
        "oco_exit_monitor starting | mode=%s | paper=%s | base=%s | poll=%ss | heartbeat=%ss | "
        "use_atr=%s | atr_lookback=%s | atr_mult_tp=%s | atr_mult_sl=%s | tp_pct=%s | sl_pct=%s | "
        "fractional_threshold=%s | fractional_tp_market_close=%s | tp_min_hold_minutes=%s | entry_lookup_days=%s | "
        "dry_run=%s | prefix=%s | error_backoff=%ss",
        mode, paper, base, poll_seconds, heartbeat_seconds,
        use_atr, atr_lookback, atr_mult_tp, atr_mult_sl, tp_pct, sl_pct,
        fractional_threshold, fractional_tp_market_close, tp_min_hold_minutes, entry_lookup_days,
        dry_run, prefix, error_backoff_seconds
    )

    tc = get_trading_client()
    dc = get_data_client()
    db_engine = get_engine()

    last_heartbeat = 0.0
    errors = 0
    last_fail_ts: Dict[str, float] = {}
    warned_fractional: Set[str] = set()
    entry_ts_cache: Dict[str, datetime] = {}
    tp_skip_log_ts: Dict[str, float] = {}

    while True:
        try:
            positions = tc.get_all_positions() or []
            open_orders = _get_orders(tc, QueryOrderStatus.OPEN)

            active_syms: Set[str] = set()
            for p in positions:
                sym = (getattr(p, "symbol", "") or "").upper()
                qty = float(getattr(p, "qty", 0) or 0)
                if qty != 0:
                    active_syms.add(sym)

            for sym in list(entry_ts_cache.keys()):
                if sym not in active_syms:
                    entry_ts_cache.pop(sym, None)

            if cleanup_orphans:
                _cleanup_orphan_exit_orders(tc, open_orders, active_syms, prefix)
                open_orders = _get_orders(tc, QueryOrderStatus.OPEN)

            protected = 0
            placed = 0
            unprotected = []

            for sym in sorted(active_syms):
                now = time.time()
                if sym in last_fail_ts and (now - last_fail_ts[sym]) < error_backoff_seconds:
                    continue

                qty, avg_entry, cur_price = _pos_info(tc, sym)
                if qty == 0:
                    continue

                qty_abs = abs(qty)
                qty_eff = qty_abs

                pos_is_long = qty > 0
                exit_side = OrderSide.SELL if pos_is_long else OrderSide.BUY
                intent = PositionIntent.SELL_TO_CLOSE if pos_is_long else PositionIntent.BUY_TO_CLOSE

                if sym not in entry_ts_cache:
                    recovered = _recover_entry_ts(
                        tc=tc,
                        db_engine=db_engine,
                        sym=sym,
                        pos_is_long=pos_is_long,
                        lookback_days=entry_lookup_days,
                        prefix=prefix,
                    )
                    if recovered:
                        entry_ts_cache[sym] = recovered

                entry_ts = entry_ts_cache.get(sym)

                is_fractional = (qty_eff > 0 and qty_eff < fractional_threshold)
                tif_eff = TimeInForce.DAY  # fractional always DAY

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
                sl = _clamp_sl_away_from_market(exit_side, sl, cur_price, min_sl_gap_pct)

                if is_fractional and sym not in warned_fractional:
                    warned_fractional.add(sym)
                    LOG.warning(
                        "fractional_mode | sym=%s qty=%.6f -> keep SL only; TP will be market-close on hit (if enabled) "
                        "AND only after tp_min_hold_minutes=%s",
                        sym, qty_eff, tp_min_hold_minutes
                    )

                # TP hit handling for fractional (market close) with min-hold
                if is_fractional and fractional_tp_market_close and _tp_hit(exit_side, cur_price, tp):
                    sl_orders = _find_sl_orders(open_orders, sym, prefix)

                    if tp_min_hold_minutes > 0:
                        if entry_ts:
                            age_min = (_utc_now() - entry_ts).total_seconds() / 60.0
                            if age_min < tp_min_hold_minutes:
                                if _should_log_throttled(tp_skip_log_ts, f"{sym}:min_hold", 60.0):
                                    LOG.info(
                                        "tp_hit_but_min_hold_active | sym=%s age_min=%.1f < %s -> skip TPMKT",
                                        sym, age_min, tp_min_hold_minutes
                                    )
                                if sl_orders:
                                    protected += 1
                                continue
                        else:
                            if _should_log_throttled(tp_skip_log_ts, f"{sym}:entry_unknown", 60.0):
                                LOG.warning(
                                    "tp_hit_but_entry_ts_unknown | sym=%s -> skip TPMKT to avoid daytrade",
                                    sym
                                )
                            if sl_orders:
                                protected += 1
                            continue

                    # cancel SL then market close
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
                        cid=f"{prefix}{sym}-TPMKT-{int(time.time())}",
                        dry_run=dry_run,
                    )
                    if ok:
                        placed += 1
                    else:
                        errors += 1
                        last_fail_ts[sym] = time.time()
                    continue

                # ensure SL exists
                sl_orders = _find_sl_orders(open_orders, sym, prefix)
                if sl_orders:
                    protected += 1
                    continue

                if not enable_repair:
                    unprotected.append(sym)
                    continue

                ok = _place_sl(
                    tc=tc,
                    sym=sym,
                    qty_abs=qty_eff,
                    exit_side=exit_side,
                    intent=intent,
                    sl=sl,
                    tif=tif_eff,
                    cid=f"{prefix}{sym}-SL-{int(time.time())}",
                    dry_run=dry_run,
                )
                if ok:
                    placed += 1
                    open_orders = _get_orders(tc, QueryOrderStatus.OPEN)
                else:
                    errors += 1
                    last_fail_ts[sym] = time.time()

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                open_orders = _get_orders(tc, QueryOrderStatus.OPEN)
                active_count = len(active_syms)
                LOG.info(
                    "Heartbeat | positions=%s protected=%s placed=%s open_orders=%s errors=%s | unprotected=%s",
                    active_count, protected, placed, len(open_orders), errors, unprotected
                )

        except Exception:
            errors += 1
            LOG.error("cycle_error", exc_info=True)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
import os
import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Set

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    LimitOrderRequest,
    StopOrderRequest,
)

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
    # If ALPACA_PAPER is explicitly set, respect it.
    if os.getenv("ALPACA_PAPER") is not None:
        return _env_bool("ALPACA_PAPER", default=True)
    return _resolve_mode() != "live"


def get_trading_client() -> TradingClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    paper = _resolve_paper()
    return TradingClient(key, secret, paper=paper)


def get_data_client() -> StockHistoricalDataClient:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return StockHistoricalDataClient(key, secret)


def _round2(x: float) -> str:
    return str(round(float(x), 2))


def _atr_pct(dc: StockHistoricalDataClient, sym: str, lookback: int = 50) -> Optional[float]:
    """
    ATR% približne z 1-min bars.
    Vracia ATR/close (napr. 0.0012 = 0.12%).
    """
    try:
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            start=None,
            end=None,
            limit=lookback,
        )
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


def _exit_leg_state(open_orders, sym: str, prefix: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Return (has_tp, has_sl, existing_base)
    existing_base = prefix + sym + "-" + <timestamp>  (bez -TP/-SL), ak už existuje aspoň jeden leg.
    """
    has_tp = False
    has_sl = False
    base = None
    symu = sym.upper()

    for o in open_orders:
        if (getattr(o, "symbol", "") or "").upper() != symu:
            continue
        cid = _cid(o)
        if not cid.startswith(prefix):
            continue
        lc = cid.lower()
        if lc.endswith("-tp"):
            has_tp = True
            base = cid[:-3]  # remove "-TP"
        if lc.endswith("-sl"):
            has_sl = True
            base = cid[:-3]  # remove "-SL"

    return has_tp, has_sl, base


def _ensure_tp_sl_order(exit_side: OrderSide, tp: float, sl: float) -> Tuple[float, float]:
    """
    Pre LONG exit (SELL): TP > SL
    Pre SHORT exit (BUY): TP < SL
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


def _pos_qty(tc: TradingClient, sym: str) -> float:
    symu = sym.upper()
    for p in (tc.get_all_positions() or []):
        if (getattr(p, "symbol", "") or "").upper() == symu:
            try:
                return float(getattr(p, "qty", 0) or 0)
            except Exception:
                return 0.0
    return 0.0


def _cancel_order_safely(tc: TradingClient, oid: str, sym: str, why: str) -> None:
    try:
        tc.cancel_order_by_id(oid)
        LOG.info("canceled_order | sym=%s oid=%s why=%s", sym, oid, why)
    except Exception as e:
        # často: already filled / already canceled
        LOG.warning("cancel_failed | sym=%s oid=%s why=%s err=%s", sym, oid, why, e)


def _cleanup_orphan_exit_orders(tc: TradingClient, open_orders, active_symbols: Set[str], prefix: str) -> int:
    """
    Zruš exit ordery (TP/SL), ktoré patria k symbolom, ktoré už nie sú v pozíciách.
    Dôležité: aby po zavretí pozície nezostal SL, ktorý by mohol spraviť short.
    """
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


def _submit_tp_sl(
    tc: TradingClient,
    sym: str,
    qty: float,
    exit_side: OrderSide,
    tp: float,
    sl: float,
    tif: TimeInForce,
    allow_ah: bool,
    cid_base: str,
    need_tp: bool,
    need_sl: bool,
    dry_run: bool,
) -> bool:
    """
    Synthetic OCO:
    1) SL FIRST (STOP) - nezvyplní sa hneď, je bezpečné to poslať prvé
    2) TP SECOND (LIMIT)
    3) Ak sa pozícia zavrie medzi krokmi, ďalší leg neposielame (aby nevznikol short).
    """
    tp, sl = _ensure_tp_sl_order(exit_side, tp, sl)

    cid_tp = f"{cid_base}-TP"
    cid_sl = f"{cid_base}-SL"

    if dry_run:
        LOG.info(
            "DRY_RUN place_exit_tp_sl | sym=%s qty=%s side=%s tp=%s sl=%s tif=%s allow_ah=%s need_tp=%s need_sl=%s cid_tp=%s cid_sl=%s",
            sym, qty, exit_side.value, _round2(tp), _round2(sl), tif.value, allow_ah, need_tp, need_sl, cid_tp, cid_sl
        )
        return True

    sl_order_id = None
    tp_order_id = None

    try:
        # 1) SL first
        if need_sl:
            sl_req = StopOrderRequest(
                symbol=sym,
                qty=float(qty),
                side=exit_side,
                time_in_force=tif,
                stop_price=_round2(sl),
                client_order_id=cid_sl,
            )
            o = tc.submit_order(sl_req)
            sl_order_id = str(getattr(o, "id", "") or "")
            LOG.info("placed_exit_sl | sym=%s qty=%s side=%s sl=%s oid=%s cid=%s", sym, qty, exit_side.value, _round2(sl), sl_order_id, cid_sl)

        # Ak sa pozícia zavrela, ďalší leg nepúšťaj (a zruš SL, ak sme ho práve vytvorili)
        if abs(_pos_qty(tc, sym)) < 1e-9:
            if sl_order_id:
                _cancel_order_safely(tc, sl_order_id, sym, "position_closed_before_tp")
            return True

        # 2) TP second
        if need_tp:
            tp_req = LimitOrderRequest(
                symbol=sym,
                qty=float(qty),
                side=exit_side,
                time_in_force=tif,
                limit_price=_round2(tp),
                client_order_id=cid_tp,
                # extended_hours dávaj len pre LIMIT a len day (ak chceš)
                extended_hours=bool(allow_ah) and tif == TimeInForce.DAY,
            )
            o = tc.submit_order(tp_req)
            tp_order_id = str(getattr(o, "id", "") or "")
            LOG.info("placed_exit_tp | sym=%s qty=%s side=%s tp=%s oid=%s cid=%s", sym, qty, exit_side.value, _round2(tp), tp_order_id, cid_tp)

        # Ak TP okamžite vyplnil a pozícia je zavretá -> zruš SL (aby nevznikol short neskôr)
        if sl_order_id and abs(_pos_qty(tc, sym)) < 1e-9:
            _cancel_order_safely(tc, sl_order_id, sym, "position_closed_after_tp")

        return True

    except Exception as e:
        msg = str(e).lower()

        # typický race: pozícia sa zavrela a SL/TP by vyzeral ako short -> ber ako OK, ale nič viac neposielaj
        if ("not allowed to short" in msg) and abs(_pos_qty(tc, sym)) < 1e-9:
            LOG.info("exit_leg_failed_but_position_closed | sym=%s err=%s", sym, e)
            return True

        LOG.error("place_exit_tp_sl failed sym=%s: %s", sym, e, exc_info=True)

        # rollback: zruš, čo sa podarilo vytvoriť (ak už nie je filled)
        try:
            if tp_order_id and not sl_order_id:
                _cancel_order_safely(tc, tp_order_id, sym, "rollback_tp_only")
            if sl_order_id and not tp_order_id:
                _cancel_order_safely(tc, sl_order_id, sym, "rollback_sl_only")
        except Exception:
            # ak je už pozícia zavretá, berieme ako OK
            if abs(_pos_qty(tc, sym)) < 1e-9:
                LOG.info("rollback_cancel_failed_but_position_closed | sym=%s", sym)
                return True
            LOG.warning("rollback_cancel_failed | sym=%s", sym, exc_info=True)

        return False


def main():
    poll_seconds = _env_int("POLL_SECONDS", 5)
    heartbeat_seconds = _env_int("HEARTBEAT_SECONDS", 60)

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
        "enable_repair=%s | dry_run=%s | cleanup_orphans=%s | prefix=%s",
        mode, paper, base, poll_seconds, heartbeat_seconds, min_qty, qty_buffer_pct,
        use_atr, atr_lookback, atr_mult_tp, atr_mult_sl, tp_pct, sl_pct, allow_ah, tif.value,
        enable_repair, dry_run, cleanup_orphans, prefix
    )

    tc = get_trading_client()
    dc = get_data_client()

    last_heartbeat = 0.0
    errors = 0

    while True:
        try:
            positions = tc.get_all_positions() or []
            open_orders = _get_open_orders(tc)

            protected = 0
            placed = 0
            repaired = 0
            unprotected = []

            active_syms = set()
            for p in positions:
                sym = (getattr(p, "symbol", "") or "").upper()
                qty = float(getattr(p, "qty", 0) or 0)
                if qty == 0:
                    continue
                active_syms.add(sym)

            # cleanup orphan exit orders (keď už nie je pozícia)
            if cleanup_orphans:
                _cleanup_orphan_exit_orders(tc, open_orders, active_syms, prefix)

                # refresh open orders after cleanup
                open_orders = _get_open_orders(tc)

            for p in positions:
                sym = (getattr(p, "symbol", "") or "").upper()
                qty = float(getattr(p, "qty", 0) or 0)
                if qty == 0:
                    continue

                qty_abs = abs(qty)
                if qty_abs < min_qty:
                    continue

                qty_eff = qty_abs * (1.0 - qty_buffer_pct / 100.0)
                if qty_eff <= 0:
                    continue

                # pozícia side: qty>0 = long, qty<0 = short
                pos_side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                exit_side = OrderSide.SELL if pos_side == OrderSide.BUY else OrderSide.BUY

                has_tp, has_sl, existing_base = _exit_leg_state(open_orders, sym, prefix=prefix)

                if has_tp and has_sl:
                    protected += 1
                    continue

                if (has_tp or has_sl) and not enable_repair:
                    protected += 1
                    continue

                unprotected.append(sym)

                avg_entry = float(getattr(p, "avg_entry_price", 0) or 0)
                ref_price = avg_entry if avg_entry > 0 else float(getattr(p, "current_price", 0) or 0)
                if ref_price <= 0:
                    continue

                # TP/SL calc
                if use_atr:
                    atrp = _atr_pct(dc, sym, lookback=atr_lookback)
                    if atrp is None:
                        # fallback to pct
                        if pos_side == OrderSide.BUY:
                            tp = ref_price * (1.0 + tp_pct / 100.0)
                            sl = ref_price * (1.0 - sl_pct / 100.0)
                        else:
                            tp = ref_price * (1.0 - tp_pct / 100.0)
                            sl = ref_price * (1.0 + sl_pct / 100.0)
                    else:
                        if pos_side == OrderSide.BUY:
                            tp = ref_price * (1.0 + atrp * atr_mult_tp)
                            sl = ref_price * (1.0 - atrp * atr_mult_sl)
                        else:
                            tp = ref_price * (1.0 - atrp * atr_mult_tp)
                            sl = ref_price * (1.0 + atrp * atr_mult_sl)
                else:
                    if pos_side == OrderSide.BUY:
                        tp = ref_price * (1.0 + tp_pct / 100.0)
                        sl = ref_price * (1.0 - sl_pct / 100.0)
                    else:
                        tp = ref_price * (1.0 - tp_pct / 100.0)
                        sl = ref_price * (1.0 + sl_pct / 100.0)

                # ak existuje už jeden leg, používaj rovnaký base, inak vytvor nový
                cid_base = existing_base if existing_base else f"{prefix}{sym}-{int(time.time())}"

                ok = _submit_tp_sl(
                    tc=tc,
                    sym=sym,
                    qty=qty_eff,
                    exit_side=exit_side,
                    tp=tp,
                    sl=sl,
                    tif=tif,
                    allow_ah=allow_ah,
                    cid_base=cid_base,
                    need_tp=(not has_tp),
                    need_sl=(not has_sl),
                    dry_run=dry_run,
                )

                if ok:
                    if has_tp or has_sl:
                        repaired += 1
                    else:
                        placed += 1
                else:
                    errors += 1

            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                open_orders = _get_open_orders(tc)
                LOG.info(
                    "Heartbeat | positions=%s protected=%s placed=%s repaired=%s open_orders=%s errors=%s | unprotected=%s",
                    len(positions), protected, placed, repaired, len(open_orders), errors, unprotected
                )

        except Exception:
            errors += 1
            LOG.error("cycle_error", exc_info=True)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
import os
import time
import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetOrderByIdRequest,
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    ReplaceOrderRequest,
)

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("oco_exit_monitor")


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
    # If ALPACA_PAPER is explicitly set, respect it (hard override).
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


def _tp_sl_from_pct(ref_price: float, position_qty: float) -> tuple[float, float]:
    tp_pct = _env_float("TP_PCT", 1.0)
    sl_pct = _env_float("SL_PCT", 0.75)
    # For long positions, tp above and sl below; for shorts we invert later by side.
    tp = ref_price * (1.0 + tp_pct / 100.0)
    sl = ref_price * (1.0 - sl_pct / 100.0)
    return tp, sl


def _atr_pct(dc: StockHistoricalDataClient, sym: str, lookback: int = 50) -> Optional[float]:
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
        log.warning("atr_pct error %s: %s", sym, e)
        return None


def _get_open_orders(tc: TradingClient, symbols: Optional[list[str]] = None):
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True, symbols=symbols)
    return tc.get_orders(req) or []


def _is_exit_order(o, prefix: str) -> bool:
    cid = (getattr(o, "client_order_id", "") or "")
    return cid.startswith(prefix)


def _place_exit_oco(
    tc: TradingClient,
    sym: str,
    qty: float,
    side: OrderSide,
    tp: float,
    sl: float,
    tif: TimeInForce,
    allow_ah: bool,
    prefix: str,
    dry_run: bool,
) -> bool:
    # For long (BUY entry), exits are SELL; for short, exits are BUY.
    exit_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

    client_oid_parent = f"{prefix}{sym}-{int(time.time())}"
    if dry_run:
        log.info("DRY_RUN place_exit_oco | sym=%s qty=%s exit_side=%s tp=%.4f sl=%.4f tif=%s allow_ah=%s cid=%s",
                 sym, qty, exit_side.value, tp, sl, tif.value, allow_ah, client_oid_parent)
        return True

    # Bracket class with take profit & stop loss legs
    try:
        req = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=exit_side,
            time_in_force=tif,
            order_class="bracket",
            take_profit=TakeProfitRequest(limit_price=str(round(tp, 2))),
            stop_loss=StopLossRequest(stop_price=str(round(sl, 2))),
            client_order_id=client_oid_parent,
            extended_hours=allow_ah,
        )
        tc.submit_order(req)
        return True
    except Exception as e:
        log.error("place_exit_oco failed sym=%s: %s", sym, e, exc_info=True)
        return False


def _needs_protection(open_orders, sym: str, prefix: str) -> bool:
    for o in open_orders:
        if (getattr(o, "symbol", "") or "").upper() != sym.upper():
            continue
        if _is_exit_order(o, prefix=prefix):
            return False
        # also treat bracket legs as protected (some APIs don't propagate CID)
        if (getattr(o, "order_class", "") or "") in ("bracket", "oco", "oto"):
            return False
    return True


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
    tif = TimeInForce(os.getenv("TIF", "day").strip().lower())

    enable_repair = _env_bool("ENABLE_REPAIR", True)
    dry_run = _env_bool("DRY_RUN", False)

    prefix = os.getenv("EXIT_PREFIX", "EXIT-OCO-").strip()

    mode = _resolve_mode()
    paper = _resolve_paper()
    base = os.getenv("ALPACA_BASE_URL")

    log.info(
        "oco_exit_monitor starting | mode=%s | paper=%s | base=%s | poll=%ss | heartbeat=%ss | min_qty=%s | qty_buffer_pct=%s | use_atr=%s | atr_lookback=%s | atr_mult_tp=%s | atr_mult_sl=%s | tp_pct=%s | sl_pct=%s | allow_ah=%s | tif=%s | enable_repair=%s | dry_run=%s | prefix=%s",
        mode, paper, base, poll_seconds, heartbeat_seconds, min_qty, qty_buffer_pct, use_atr, atr_lookback, atr_mult_tp, atr_mult_sl, tp_pct, sl_pct, allow_ah, tif.value, enable_repair, dry_run, prefix
    )

    tc = get_trading_client()
    dc = get_data_client()

    last_heartbeat = 0.0
    counts = defaultdict(int)
    errors = 0

    while True:
        try:
            positions = tc.get_all_positions() or []
            open_orders = _get_open_orders(tc)

            protected = 0
            placed = 0
            repaired = 0
            skipped_closing = 0
            unprotected = []

            for p in positions:
                sym = (getattr(p, "symbol", "") or "").upper()
                qty = float(getattr(p, "qty", 0) or 0)
                if qty == 0:
                    continue

                side = OrderSide.BUY if qty > 0 else OrderSide.SELL
                qty_abs = abs(qty)
                if qty_abs < min_qty:
                    continue

                # buffer qty down a bit if desired
                qty_eff = qty_abs * (1.0 - qty_buffer_pct / 100.0)
                if qty_eff <= 0:
                    continue

                if not _needs_protection(open_orders, sym, prefix=prefix):
                    protected += 1
                    continue

                unprotected.append(sym)

                # reference price = avg entry price (fallback to current price if missing)
                avg_entry = float(getattr(p, "avg_entry_price", 0) or 0)
                ref_price = avg_entry if avg_entry > 0 else float(getattr(p, "current_price", 0) or 0)
                if ref_price <= 0:
                    continue

                # compute tp/sl either ATR-based or pct-based
                if use_atr:
                    atrp = _atr_pct(dc, sym, lookback=atr_lookback)
                    if atrp is None:
                        tp, sl = _tp_sl_from_pct(ref_price, qty_eff)
                    else:
                        if side == OrderSide.BUY:
                            tp = ref_price * (1.0 + atrp * atr_mult_tp)
                            sl = ref_price * (1.0 - atrp * atr_mult_sl)
                        else:
                            tp = ref_price * (1.0 - atrp * atr_mult_tp)
                            sl = ref_price * (1.0 + atrp * atr_mult_sl)
                else:
                    if side == OrderSide.BUY:
                        tp = ref_price * (1.0 + tp_pct / 100.0)
                        sl = ref_price * (1.0 - sl_pct / 100.0)
                    else:
                        tp = ref_price * (1.0 - tp_pct / 100.0)
                        sl = ref_price * (1.0 + sl_pct / 100.0)

                ok = _place_exit_oco(
                    tc=tc,
                    sym=sym,
                    qty=qty_eff,
                    side=side,
                    tp=tp,
                    sl=sl,
                    tif=tif,
                    allow_ah=allow_ah,
                    prefix=prefix,
                    dry_run=dry_run,
                )
                if ok:
                    placed += 1
                else:
                    errors += 1

            # heartbeat
            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                log.info(
                    "Heartbeat | positions=%s protected=%s placed=%s repaired=%s skipped_closing=%s open_orders=%s errors=%s | unprotected=%s",
                    len(positions), protected, placed, repaired, skipped_closing, len(open_orders), errors, unprotected
                )

        except Exception:
            errors += 1
            log.error("cycle_error", exc_info=True)

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()

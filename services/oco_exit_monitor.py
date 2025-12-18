# services/oco_exit_monitor.py
import os
import time
import logging
from typing import Optional

from alpaca.trading.client import TradingClient

from services.alpaca_exit_guard import (
    has_exit_orders,
    cancel_exit_orders,
    place_exit_oco,
    round_to_step,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("oco_exit_monitor")


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _get_env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return default


def _symbols():
    raw = (os.getenv("SYMBOLS") or "AAPL,MSFT,SPY").strip()
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _safe_get_position(tc: TradingClient, symbol: str):
    try:
        return tc.get_open_position(symbol)
    except Exception:
        return None


def main():
    tc = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_API_SECRET"), paper=True)

    SYMBOLS = _symbols()
    POLL_SEC = _get_env_int("OCO_POLL_SEC", 20)

    ATR_PCT = _get_env_float("ATR_PCT", 0.01)      # napr. 0.01 = 1%
    TP_MULT = _get_env_float("TP_MULT", 1.75)
    SL_MULT = _get_env_float("SL_MULT", 1.75)

    PRICE_STEP = _get_env_float("PRICE_STEP", 0.01)

    ORPHAN_MIN_AGE_SEC = _get_env_int("ORPHAN_MIN_AGE_SEC", 120)
    CANCEL_ORPHAN = _get_env_bool("CANCEL_ORPHAN", True)

    DRY_RUN = _get_env_bool("DRY_RUN", False)

    log.info(
        "oco_exit_monitor starting | SYMBOLS=%s | POLL=%ss | ATR_PCT=%.4f | TP_MULT=%.2f | SL_MULT=%.2f | "
        "PRICE_STEP=%.2f | ORPHAN_MIN_AGE_SEC=%s | CANCEL_ORPHAN=%s | DRY_RUN=%s",
        SYMBOLS, POLL_SEC, ATR_PCT, TP_MULT, SL_MULT, PRICE_STEP, ORPHAN_MIN_AGE_SEC, CANCEL_ORPHAN, DRY_RUN
    )

    while True:
        try:
            for symbol in SYMBOLS:
                pos = _safe_get_position(tc, symbol)

                # ===== no position -> orphan cleanup (voliteľne) =====
                if not pos:
                    if CANCEL_ORPHAN and has_exit_orders(tc, symbol):
                        # jednoduché: necháme to ako bezpečnostný “kill-switch”
                        log.warning("Orphan EXIT detected | %s | canceling (DRY_RUN=%s)", symbol, DRY_RUN)
                        cancel_exit_orders(tc, symbol, dry_run=DRY_RUN)
                    continue

                # ===== position exists -> ensure exactly one EXIT =====
                qty_raw = float(pos.qty)  # pre SHORT môže byť záporné
                avg = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)

                if avg <= 0 or qty_raw == 0:
                    log.warning("Invalid position data | %s | qty=%s avg=%s", symbol, qty_raw, avg)
                    continue

                # ak už existuje OCO (parent), nerob duplicitné
                if has_exit_orders(tc, symbol):
                    log.info("EXIT already exists | %s | skip", symbol)
                    continue

                # ===== compute TP/SL =====
                # ATR-based delta as percent of avg
                delta = avg * ATR_PCT

                # LONG: TP above, SL below
                # SHORT: TP below, SL above
                direction = 1.0 if qty_raw > 0 else -1.0

                tp = avg + direction * delta * TP_MULT
                sl = avg - direction * delta * SL_MULT

                tp = round_to_step(tp, PRICE_STEP)
                sl = round_to_step(sl, PRICE_STEP)

                log.info(
                    "Placing EXIT-OCO | %s | qty=%s avg=%.4f -> TP=%.4f SL=%.4f | DRY_RUN=%s",
                    symbol, qty_raw, avg, tp, sl, DRY_RUN
                )

                if not DRY_RUN:
                    oid = place_exit_oco(tc, symbol, qty_raw, tp_price=tp, sl_stop_price=sl)
                    log.info("EXIT-OCO submitted | %s | order_id=%s", symbol, oid)

            time.sleep(POLL_SEC)

        except Exception:
            # toto je dôležité, aby container “neumrel” pri jednej API chybe
            log.exception("Cycle error (will continue)")
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()

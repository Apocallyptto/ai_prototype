# services/signal_executor.py

import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from alpaca.trading.client import TradingClient

# --- Ensure project root (/app) is on sys.path inside Docker ---
ROOT = Path(__file__).resolve().parent.parent  # /app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# risk layer imports
from jobs.risk_limits import (
    load_limits_from_env,
    compute_qty_for_long,
    can_open_new_position,
)

# Order router (your existing one)
from services.order_router import place_entry


log = logging.getLogger("signal_executor")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)

# -----------------------
# ENV
# -----------------------

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")


def _env_bool(name: str, default_true: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default_true
    return str(v).strip() not in ("0", "false", "False", "no", "No")


SYMBOLS: List[str] = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",")
    if s.strip()
]

PORTFOLIO_ID: Optional[str] = os.getenv("PORTFOLIO_ID") or None
MIN_STRENGTH: float = float(os.getenv("MIN_STRENGTH", "0.60"))
POLL_SECONDS: int = int(os.getenv("SIGNAL_POLL_SECONDS", "20"))

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = _env_bool("ALPACA_PAPER", default_true=True)

# Load risk limits at startup
RISK_LIMITS = load_limits_from_env()


# -----------------------
# DB Helpers
# -----------------------

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    return conn


def fetch_unprocessed_signals(conn) -> List[Dict[str, Any]]:
    q = """
    SELECT id, created_at, symbol, side, strength, portfolio_id, status
    FROM signals
    WHERE
        status = 'pending'
        AND processed_at IS NULL
        AND strength >= %s
        AND symbol = ANY(%s)
        AND (%s IS NULL OR portfolio_id = %s)
    ORDER BY created_at ASC
    LIMIT 20;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q, (MIN_STRENGTH, SYMBOLS, PORTFOLIO_ID, PORTFOLIO_ID))
        rows = cur.fetchall()
        return list(rows)


def mark_signal(
    conn,
    signal_id: int,
    status: str,
    error: Optional[str] = None,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    exec_order_id: Optional[str] = None,
) -> None:
    q = """
    UPDATE signals
    SET
        status = %s,
        processed_at = NOW(),
        error = %s,
        order_id = %s,
        client_order_id = %s,
        exec_order_id = %s
    WHERE id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(
            q,
            (status, error, order_id, client_order_id, exec_order_id, signal_id),
        )


# -----------------------
# Order ID extractor
# -----------------------

def _extract_ids(o: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if o is None:
        return (None, None, None)

    # Alpaca Order object
    if hasattr(o, "id"):
        return (
            str(getattr(o, "id", None)) if getattr(o, "id", None) is not None else None,
            str(getattr(o, "client_order_id", None))
            if getattr(o, "client_order_id", None) is not None
            else None,
            None,
        )

    # dict
    if isinstance(o, dict):
        oid = o.get("id")
        coid = o.get("client_order_id")
        xoid = o.get("exec_order_id")
        return (
            str(oid) if oid is not None else None,
            str(coid) if coid is not None else None,
            str(xoid) if xoid is not None else None,
        )

    # tuple/list
    if isinstance(o, (tuple, list)):
        oid = o[0] if len(o) > 0 else None
        coid = o[1] if len(o) > 1 else None
        xoid = o[2] if len(o) > 2 else None
        return (
            str(oid) if oid is not None else None,
            str(coid) if coid is not None else None,
            str(xoid) if xoid is not None else None,
        )

    # fallback
    try:
        return (
            str(getattr(o, "id", None)) if getattr(o, "id", None) is not None else None,
            str(getattr(o, "client_order_id", None))
            if getattr(o, "client_order_id", None) is not None
            else None,
            None,
        )
    except Exception:
        return (None, None, None)


# -----------------------
# Risk-aware processing
# -----------------------

def process_one(tc: TradingClient, sig: Dict[str, Any], conn) -> None:
    symbol = sig["symbol"]
    side = sig["side"]
    signal_id = sig["id"]

    # 1) max open positions guard
    open_positions = tc.get_all_positions()
    if not can_open_new_position(len(open_positions), RISK_LIMITS):
        log.info(
            "risk_guard: max_open_positions=%s reached, SKIPPING signal id=%s symbol=%s",
            RISK_LIMITS.max_open_positions,
            signal_id,
            symbol,
        )
        mark_signal(conn, signal_id, status="skipped_risk", error="max_open_positions")
        return

    # 2) account equity
    account = tc.get_account()
    try:
        equity = float(account.equity)
    except Exception:
        equity = 0.0

    # 3) entry price (simple: last trade)
    last_trade = tc.get_last_trade(symbol)
    entry_px = float(last_trade.price)

    # TODO: neskôr nahradíme skutočným ATR/SL/TP z tvojich ATR nástrojov / DB
    # Teraz iba placeholder, aby risk vrstvy fungovali.
    atr = 0.5
    sl_px = entry_px - 1.0 * atr
    tp_px = entry_px + 1.5 * atr

    # 4) risk-based sizing
    qty = compute_qty_for_long(
        entry_price=entry_px,
        stop_loss_price=sl_px,
        equity=equity,
        limits=RISK_LIMITS,
    )

    if qty <= 0:
        log.info(
            "risk_guard: qty=0, SKIPPING signal id=%s symbol=%s entry=%.2f sl=%.2f equity=%.2f",
            signal_id,
            symbol,
            entry_px,
            sl_px,
            equity,
        )
        mark_signal(conn, signal_id, status="skipped_risk", error="qty_zero")
        return

    # 5) place order via router
    client_id = str(uuid.uuid4())
    try:
        o = place_entry(
            tc,
            symbol=symbol,
            side=side,
            use_limit=False,
            qty=qty,
            client_order_id=client_id,
            # NOTE: if your router supports tp/sl, you can pass them here too
            # tp_price=tp_px,
            # sl_price=sl_px,
        )
    except TypeError:
        # fallback if router does not accept client_order_id
        o = place_entry(
            tc,
            symbol=symbol,
            side=side,
            use_limit=False,
            qty=qty,
        )

    oid, coid, xoid = _extract_ids(o)

    log.info(
        "submitted with risk: %s %s | qty=%s | entry=%.2f | tp=%.2f | sl=%.2f | oid=%s client_order_id=%s",
        symbol,
        side,
        qty,
        entry_px,
        tp_px,
        sl_px,
        oid,
        coid,
    )

    if oid is None and coid is None:
        raise RuntimeError("broker returned no order IDs")

    mark_signal(
        conn,
        signal_id,
        status="submitted",
        error=None,
        order_id=oid,
        client_order_id=coid,
        exec_order_id=xoid,
    )


# -----------------------
# Main loop
# -----------------------

def loop(tc: TradingClient, conn) -> None:
    log.info(
        "signal_executor starting | MIN_STRENGTH=%.2f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss | RISK=%s",
        MIN_STRENGTH,
        ",".join(SYMBOLS),
        PORTFOLIO_ID or "",
        POLL_SECONDS,
        RISK_LIMITS,
    )
    while True:
        try:
            sigs = fetch_unprocessed_signals(conn)
            if not sigs:
                log.info("no new signals")
                time.sleep(POLL_SECONDS)
                continue

            for sig in sigs:
                try:
                    process_one(tc, sig, conn)
                except Exception as e:
                    err = str(e)
                    log.warning(
                        "signal %s %s failed: %s", sig["symbol"], sig["side"], err
                    )
                    try:
                        mark_signal(
                            conn,
                            sig["id"],
                            status="error",
                            error=err,
                            order_id=None,
                            client_order_id=None,
                            exec_order_id=None,
                        )
                    except Exception as inner:
                        log.error(
                            "failed to mark error for signal id=%s: %s",
                            sig["id"],
                            inner,
                        )
        except Exception as e:
            log.error("main loop error: %s", e)
            time.sleep(POLL_SECONDS)


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET must be set")

    tc = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=ALPACA_PAPER)
    conn = get_conn()
    try:
        loop(tc, conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

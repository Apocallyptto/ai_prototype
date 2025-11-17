# services/signal_executor.py

import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
from alpaca.trading.client import TradingClient

# risk layer imports
from jobs.risk_limits import (
    load_limits_from_env,
    compute_qty_for_long,
    can_open_new_position,
)

# Order router (your existing)
from services.order_router import place_entry

log = logging.getLogger("signal_executor")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%message)s",
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

SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

PORTFOLIO_ID: Optional[str] = os.getenv("PORTFOLIO_ID") or None
MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
POLL_SECONDS = int(os.getenv("SIGNAL_POLL_SECONDS", "20"))

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = _env_bool("ALPACA_PAPER", default_true=True)

# Load risk limits at start
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
        return list(cur.fetchall())


def mark_signal(conn, signal_id: int, status: str,
                error: Optional[str] = None,
                order_id: Optional[str] = None,
                client_order_id: Optional[str] = None,
                exec_order_id: Optional[str] = None) -> None:
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
        cur.execute(q, (status, error, order_id, client_order_id, exec_order_id, signal_id))


# -----------------------
# Order ID extractor
# -----------------------

def _extract_ids(o: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if o is None:
        return (None, None, None)
    if hasattr(o, "id"):
        return (
            str(getattr(o, "id", None)),
            str(getattr(o, "client_order_id", None)),
            None,
        )
    if isinstance(o, dict):
        return (
            str(o.get("id")) if o.get("id") else None,
            str(o.get("client_order_id")) if o.get("client_order_id") else None,
            str(o.get("exec_order_id")) if o.get("exec_order_id") else None,
        )
    if isinstance(o, (tuple, list)):
        oid = o[0] if len(o) > 0 else None
        coid = o[1] if len(o) > 1 else None
        xoid = o[2] if len(o) > 2 else None
        return (str(oid), str(coid), str(xoid))
    try:
        return (
            str(getattr(o, "id", None)),
            str(getattr(o, "client_order_id", None)),
            None,
        )
    except Exception:
        return (None, None, None)


# -----------------------
# Risk-aware execution
# -----------------------

def process_one(tc: TradingClient, sig: Dict[str, Any], conn) -> None:
    symbol = sig["symbol"]
    side = sig["side"]
    signal_id = sig["id"]

    # 1) Check max open positions
    open_positions = tc.get_all_positions()
    if not can_open_new_position(len(open_positions), RISK_LIMITS):
        log.info(
            "risk_guard: max_open_positions reached (%s). SKIPPING signal id=%s symbol=%s",
            RISK_LIMITS.max_open_positions,
            signal_id,
            symbol
        )
        mark_signal(conn, signal_id, status="skipped_risk", error="max_open_positions")
        return

    # 2) Get account equity
    account = tc.get_account()
    try:
        equity = float(account.equity)
    except Exception:
        equity = 0.0

    # 3) Get market data for entry price (use quote or last trade)
    # Prefer your existing order router behaviour for entry.
    # But we need entry + SL → so default use current market price:
    last_px = float(tc.get_last_trade(symbol).price)

    # Hard-coded basic ATR exit logic (SAMPLE - refine in your own broker logic)
    atr = 0.5  # placeholder if you don't have ATR in signals table
    sl_price = last_px - atr * 1.0
    tp_price = last_px + atr * 1.5

    # 4) Compute safe qty
    qty = compute_qty_for_long(
        entry_price=last_px,
        stop_loss_price=sl_price,
        equity=equity,
        limits=RISK_LIMITS,
    )

    if qty <= 0:
        log.info(
            "risk_guard: qty=0 → SKIPPING signal id=%s symbol=%s entry=%.2f sl=%.2f equity=%.2f",
            signal_id, symbol, last_px, sl_price, equity
        )
        mark_signal(conn, signal_id, status="skipped_risk", error="qty_zero")
        return

    # 5) Place the order through your router
    client_id = str(uuid.uuid4())
    try:
        o = place_entry(
            tc,
            symbol=symbol,
            side=side,
            use_limit=False,
            qty=qty,
            client_order_id=client_id,
        )
    except TypeError:
        o = place_entry(
            tc,
            symbol=symbol,
            side=side,
            use_limit=False,
            qty=qty
        )

    oid, coid, xoid = _extract_ids(o)

    log.info(
        "submitted with risk: %s %s | qty=%s | entry=%.2f | tp=%.2f | sl=%.2f | oid=%s",
        symbol, side, qty, last_px, tp_price, sl_price, oid
    )

    if oid is None and coid is None:
        raise RuntimeError("No order IDs returned from broker")

    # mark DB
    mark_signal(conn, signal_id, status="submitted",
                error=None, order_id=oid, client_order_id=coid, exec_order_id=xoid)


# -----------------------
# Main loop
# -----------------------

def loop(tc: TradingClient, conn) -> None:
    log.info(
        "signal_executor starting | MIN_STRENGTH=%.2f | SYMBOLS=%s | RISK=%s",
        MIN_STRENGTH, ",".join(SYMBOLS), RISK_LIMITS
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
                    log.error("signal %s %s failed: %s", sig["symbol"], sig["side"], err)
                    try:
                        mark_signal(conn, sig["id"], status="error",
                                    error=err, order_id=None, client_order_id=None, exec_order_id=None)
                    except:
                        pass
        except Exception as e:
            log.error("main loop error: %s", e)
            time.sleep(POLL_SECONDS)


def main():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Missing ALPACA credentials")

    tc = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=ALPACA_PAPER)
    conn = get_conn()
    try:
        loop(tc, conn)
    finally:
        try:
            conn.close()
        except:
            pass


if __name__ == "__main__":
    main()

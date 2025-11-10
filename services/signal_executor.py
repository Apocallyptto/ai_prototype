# services/signal_executor.py

import os
import time
import logging
from typing import Dict, Any, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from alpaca.trading.client import TradingClient

# Our order entry helper (already in your repo)
from services.order_router import place_entry


# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("signal_executor")


# -----------------------------
# Env/config helpers
# -----------------------------
def env_bool(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default)
    return str(v).strip() not in ("0", "", "false", "False", "FALSE", "no", "No", "NO")


DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER = env_bool("ALPACA_PAPER", "1")

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
SYMBOLS_RAW = os.getenv("SYMBOLS", "")  # comma-separated
SYMBOLS_SET = {s.strip().upper() for s in SYMBOLS_RAW.split(",") if s.strip()} if SYMBOLS_RAW else set()

PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")  # optional filter
SIGNAL_POLL_SECONDS = int(os.getenv("SIGNAL_POLL_SECONDS", "20"))


def _sizing_kwargs() -> Dict[str, Any]:
    """
    Prefer NOTIONAL_USD if present and > 0, otherwise use FIXED_QTY if present and > 0.
    Fallback to qty=1.0. This avoids passing an unsupported 'notional' arg when env is blank.
    """
    notional_env = os.getenv("NOTIONAL_USD")
    qty_env = os.getenv("FIXED_QTY")
    try:
        n = float(notional_env) if (notional_env is not None and notional_env.strip() != "") else 0.0
    except ValueError:
        n = 0.0
    try:
        q = float(qty_env) if (qty_env is not None and qty_env.strip() != "") else 0.0
    except ValueError:
        q = 0.0

    if n > 0:
        return {"notional": n}
    if q > 0:
        return {"qty": q}
    return {"qty": 1.0}


# -----------------------------
# DB helpers
# -----------------------------
def get_conn():
    return psycopg2.connect(DB_URL)


def fetch_unprocessed_signals(conn) -> List[Dict[str, Any]]:
    """
    Fetch signals to execute:
      - status IS NULL OR status='pending'
      - processed_at IS NULL
      - strength >= MIN_STRENGTH
      - optional SYMBOLS filter
      - optional PORTFOLIO_ID filter
    Oldest first.
    """
    filters = [
        "(status IS NULL OR status = 'pending')",
        "processed_at IS NULL",
        "strength >= %s",
    ]
    params: List[Any] = [MIN_STRENGTH]

    if SYMBOLS_SET:
        sym_list = sorted(SYMBOLS_SET)
        placeholders = ", ".join(["%s"] * len(sym_list))
        filters.append(f"UPPER(symbol) IN ({placeholders})")
        params.extend(sym_list)

    if PORTFOLIO_ID and PORTFOLIO_ID.strip():
        filters.append("COALESCE(portfolio_id, '') = %s")
        params.append(PORTFOLIO_ID.strip())

    where_clause = " AND ".join(filters)
    q = f"""
        SELECT id, created_at, symbol, side, strength, source, portfolio_id
        FROM signals
        WHERE {where_clause}
        ORDER BY created_at ASC
        LIMIT 20
    """

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    return rows


def mark_signal(
    conn,
    signal_id: int,
    status: str,
    error: Optional[str] = None,
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    exec_order_id: Optional[str] = None,
):
    """
    Update a signal row with processing result. All optional fields are written
    using NULL when missing.
    """
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
    conn.commit()


# -----------------------------
# Execution
# -----------------------------
def build_trading_client() -> TradingClient:
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        log.warning("ALPACA_API_KEY/SECRET not set; TradingClient will likely fail for live API calls.")
    return TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)


def process_one(tc: TradingClient, sig: Dict[str, Any], conn) -> None:
    sid = sig["id"]
    symbol = str(sig["symbol"]).upper()
    side = str(sig["side"]).lower()

    # Sanity
    if side not in ("buy", "sell"):
        err = f"invalid side '{side}'"
        log.warning("signal %s %s failed: %s", symbol, side, err)
        mark_signal(conn, sid, status="error", error=err)
        return

    try:
        kwargs = _sizing_kwargs()
        # IMPORTANT: do not pass 'notional' unless _sizing_kwargs chose it
        # place_entry signature supports: (tc, symbol, side, qty=?, notional=?, use_limit=?)
        o = place_entry(
            tc,
            symbol=symbol,
            side=side,
            use_limit=False,
            **kwargs,
        )

        order_id = getattr(o, "id", None)
        client_order_id = getattr(o, "client_order_id", None)

        # If you generate your own exec order id (router), populate it; else leave None
        exec_order_id = None

        log.info("signal %s %s submitted: order_id=%s client_order_id=%s kwargs=%s",
                 symbol, side, order_id, client_order_id, kwargs)

        mark_signal(
            conn,
            sid,
            status="submitted",
            error=None,
            order_id=order_id,
            client_order_id=client_order_id,
            exec_order_id=exec_order_id,
        )

    except TypeError as te:
        # Typical case when someone passed an unsupported kwarg (e.g., 'notional')
        err = str(te)
        log.warning("signal %s %s failed: %s", symbol, side, err)
        mark_signal(conn, sid, status="error", error=err)

    except Exception as e:
        err = str(e)
        log.warning("signal %s %s failed: %s", symbol, side, err)
        mark_signal(conn, sid, status="error", error=err)


def loop():
    tc = build_trading_client()

    with get_conn() as conn:
        while True:
            try:
                sigs = fetch_unprocessed_signals(conn)
                if not sigs:
                    log.info("no new signals")
                    time.sleep(SIGNAL_POLL_SECONDS)
                    continue

                for sig in sigs:
                    process_one(tc, sig, conn)

                # small breather to avoid hammering
                time.sleep(1)

            except Exception as e:
                log.error("main loop error: %s", e, exc_info=True)
                # brief backoff on unexpected failures
                time.sleep(min(SIGNAL_POLL_SECONDS, 10))


def main():
    log.info("signal_executor starting | MIN_STRENGTH=%.2f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss",
             MIN_STRENGTH, ",".join(sorted(SYMBOLS_SET)) if SYMBOLS_SET else "(all)", PORTFOLIO_ID, SIGNAL_POLL_SECONDS)
    loop()


if __name__ == "__main__":
    main()

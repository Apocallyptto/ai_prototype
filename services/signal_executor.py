# services/signal_executor.py
import os
import time
import logging
import psycopg2
import psycopg2.extras

from alpaca.trading.client import TradingClient
from services.order_router import place_entry  # absolute import

log = logging.getLogger("signal_executor")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s"
)

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise RuntimeError("DB_URL is not set")

MIN_STRENGTH = float(os.getenv("MIN_STRENGTH", "0.60"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "").split(",") if s.strip()]
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")  # optional
POLL = int(os.getenv("SIGNAL_POLL_SECONDS", "20"))

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
PAPER = os.getenv("ALPACA_PAPER", "1") != "0"

def _sizing_kwargs():
    """
    Prefer NOTIONAL_USD if present and > 0, otherwise use FIXED_QTY if present and > 0.
    Fallback to qty=1.0.
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

def _s(v):
    """Safely stringify values for DB (handles UUID objects)."""
    return str(v) if v is not None else None

def connect():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True  # <- important: no transaction nesting issues
    return conn

def fetch_unprocessed_signals(conn):
    symbol_filter = ""
    params = [str(MIN_STRENGTH)]

    if PORTFOLIO_ID:
        portfolio_filter = "AND (portfolio_id = %s OR portfolio_id IS NULL)"
        params.append(PORTFOLIO_ID)
    else:
        portfolio_filter = ""

    if SYMBOLS:
        placeholders = ",".join(["%s"] * len(SYMBOLS))
        symbol_filter = f"AND symbol = ANY(ARRAY[{placeholders}])"
        params.extend(SYMBOLS)

    q = f"""
        SELECT id, symbol, side, strength, portfolio_id
        FROM signals
        WHERE status = 'pending'
          AND processed_at IS NULL
          AND strength >= %s
          {portfolio_filter}
          {symbol_filter}
        ORDER BY created_at ASC
        LIMIT 20;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(q, params)
        return cur.fetchall()

def mark_signal(conn, signal_id, status, error=None, order_id=None, client_order_id=None, exec_order_id=None):
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
    # DO NOT re-enter the connection context; just open a cursor and let autocommit handle it
    with conn.cursor() as cur:
        cur.execute(q, (_s(status), _s(error), _s(order_id), _s(client_order_id), _s(exec_order_id), signal_id))

def process_one(tc, sig, conn):
    try:
        kwargs = _sizing_kwargs()
        o = place_entry(
            tc,
            symbol=sig["symbol"],
            side=sig["side"],
            use_limit=False,
            **kwargs,
        )

        oid  = _s(getattr(o, "id", None))
        coid = _s(getattr(o, "client_order_id", None))
        xoid = None  # fill later if you wire exec IDs

        log.info(
            "signal %s %s submitted: order_id=%s client_order_id=%s kwargs=%s",
            sig["symbol"], sig["side"], oid, coid, kwargs
        )

        # If broker returned nothing, treat as error to avoid leaving 'pending' forever
        if oid is None and coid is None:
            raise RuntimeError("broker returned no order IDs (check API keys/market status)")

        mark_signal(conn, sig["id"], status="submitted", error=None, order_id=oid, client_order_id=coid, exec_order_id=xoid)

    except Exception as e:
        err = f"{e}"
        log.warning("signal %s %s failed: %s", sig["symbol"], sig["side"], err)
        mark_signal(conn, sig["id"], status="error", error=err, order_id=None, client_order_id=None, exec_order_id=None)

def loop():
    log.info(
        "signal_executor starting | MIN_STRENGTH=%.2f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss",
        MIN_STRENGTH, ",".join(SYMBOLS) if SYMBOLS else "(all)", PORTFOLIO_ID or "(any)", POLL
    )
    tc = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=PAPER)

    conn = None
    while True:
        try:
            if conn is None or conn.closed:
                conn = connect()

            sigs = fetch_unprocessed_signals(conn)
            if not sigs:
                log.info("no new signals")
            for sig in sigs:
                process_one(tc, sig, conn)

        except Exception as e:
            log.error("main loop error: %s", e, exc_info=True)
            # if DB connection is the problem, drop it and reconnect next tick
            try:
                if conn and not conn.closed:
                    conn.close()
            except Exception:
                pass
            conn = None

        time.sleep(POLL)

if __name__ == "__main__":
    loop()

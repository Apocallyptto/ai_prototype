# services/signal_executor.py

import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras

from alpaca.trading.client import TradingClient

# NOTE: we only *import the function signature* you already have.
# Ensure services/order_router.py exports place_entry(tc, symbol, side, use_limit=False, **kwargs)
from services.order_router import place_entry

log = logging.getLogger("signal_executor")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
)

# -----------------------
# Env helpers / constants
# -----------------------

def _env_bool(name: str, default_true: bool = True) -> bool:
    v = os.getenv(name)
    if v is None:
        return default_true
    return str(v).strip() not in ("0", "false", "False", "no", "No")


DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

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


# -----------------------
# Sizing & ID extraction
# -----------------------

def _sizing_kwargs() -> Dict[str, Any]:
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


def _extract_ids(o: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (order_id, client_order_id, exec_order_id) from various return shapes."""
    if o is None:
        return (None, None, None)
    # Alpaca Order object
    if hasattr(o, "id"):
        return (
            str(getattr(o, "id", None)) if getattr(o, "id", None) is not None else None,
            str(getattr(o, "client_order_id", None)) if getattr(o, "client_order_id", None) is not None else None,
            None,
        )
    # dict-ish
    if isinstance(o, dict):
        oid = o.get("id")
        coid = o.get("client_order_id")
        xoid = o.get("exec_order_id")
        return (str(oid) if oid is not None else None,
                str(coid) if coid is not None else None,
                str(xoid) if xoid is not None else None)
    # tuple/list: (id, client_order_id[, exec_id])
    if isinstance(o, (tuple, list)):
        oid  = o[0] if len(o) > 0 else None
        coid = o[1] if len(o) > 1 else None
        xoid = o[2] if len(o) > 2 else None
        return (str(oid) if oid is not None else None,
                str(coid) if coid is not None else None,
                str(xoid) if xoid is not None else None)
    # Fallback best effort
    try:
        return (
            str(getattr(o, "id", None)) if getattr(o, "id", None) is not None else None,
            str(getattr(o, "client_order_id", None)) if getattr(o, "client_order_id", None) is not None else None,
            None,
        )
    except Exception:
        return (None, None, None)


# -----------------------
# DB helpers
# -----------------------

def get_conn():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True  # avoids "connection cannot be re-entered recursively"
    return conn


def fetch_unprocessed_signals(conn) -> List[Dict[str, Any]]:
    """
    Pull newest 'pending' signals that match filters and weren't processed yet.
    """
    # Symbol filter as array for performance; strength gating; optional portfolio gating
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


def mark_signal(conn, signal_id: int, status: str,
                error: Optional[str] = None,
                order_id: Optional[str] = None,
                client_order_id: Optional[str] = None,
                exec_order_id: Optional[str] = None) -> None:
    """
    Update a signal row; all optional fields are nullable-safe.
    NOTE: Do NOT use `with conn, conn.cursor()` since autocommit=True; just use cursor.
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


# -----------------------
# Core processing
# -----------------------

def process_one(tc: TradingClient, sig: Dict[str, Any], conn) -> None:
    kwargs = _sizing_kwargs()
    client_id = str(uuid.uuid4())

    # Try to pass client_order_id if router supports it; fall back if not.
    try:
        o = place_entry(
            tc,
            symbol=sig["symbol"],
            side=sig["side"],
            use_limit=False,
            client_order_id=client_id,
            **kwargs,
        )
    except TypeError:
        o = place_entry(
            tc,
            symbol=sig["symbol"],
            side=sig["side"],
            use_limit=False,
            **kwargs,
        )

    oid, coid, xoid = _extract_ids(o)
    log.info(
        "signal %s %s submitted: order_id=%s client_order_id=%s kwargs=%s",
        sig["symbol"], sig["side"], oid, coid, kwargs
    )

    # If broker returned nothing, fail fast so the row doesn't stay pending forever
    if oid is None and coid is None:
        raise RuntimeError("broker returned no order IDs (check API keys/market status)")

    mark_signal(conn, sig["id"], status="submitted",
                error=None, order_id=oid, client_order_id=coid, exec_order_id=xoid)


def loop(tc: TradingClient, conn) -> None:
    log.info(
        "signal_executor starting | MIN_STRENGTH=%.2f | SYMBOLS=%s | PORTFOLIO_ID=%s | POLL=%ss",
        MIN_STRENGTH, ",".join(SYMBOLS), PORTFOLIO_ID or "", POLL_SECONDS
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
                    log.warning("signal %s %s failed: %s", sig["symbol"], sig["side"], err)
                    # On error, mark row as error (don’t re-enter connection context; autocommit handles it)
                    try:
                        mark_signal(conn, sig["id"], status="error",
                                    error=err, order_id=None, client_order_id=None, exec_order_id=None)
                    except Exception as inner:
                        log.error("failed to mark error for signal id=%s: %s", sig["id"], inner)
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

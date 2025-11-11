# --- keep your existing imports ---
import os
import logging
from alpaca.trading.client import TradingClient

from services.order_router import place_entry

# ... your other imports ...

log = logging.getLogger("signal_executor")

# --- sizing helper (as you already added) ---
def _sizing_kwargs():
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
    return str(v) if v is not None else None

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
    with conn, conn.cursor() as cur:
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
        xoid = None  # set if you derive exec id elsewhere

        log.info(
            "signal %s %s submitted: order_id=%s client_order_id=%s kwargs=%s",
            sig["symbol"], sig["side"], oid, coid, kwargs
        )
        mark_signal(conn, sig["id"], status="submitted", error=None, order_id=oid, client_order_id=coid, exec_order_id=xoid)

    except Exception as e:
        err = f"{e}"
        log.warning("signal %s %s failed: %s", sig["symbol"], sig["side"], err)
        mark_signal(conn, sig["id"], status="error", error=err, order_id=None, client_order_id=None, exec_order_id=None)

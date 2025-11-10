import os
import time
import uuid
import logging
from decimal import Decimal

import psycopg2
import psycopg2.extras

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from services.order_router import place_entry  # we rely on your router’s safety

log = logging.getLogger("signal_executor")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ----------------
# Env & helpers
# ----------------

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v not in ("0", "false", "False", "no", "NO")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

MIN_STRENGTH = Decimal(os.getenv("MIN_STRENGTH", "0.60"))
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "AAPL,MSFT,SPY").split(",") if s.strip()]

# Sizing: choose one of the knobs below
FIXED_QTY = Decimal(os.getenv("FIXED_QTY", "0.05"))       # fractional shares default
NOTIONAL_USD = Decimal(os.getenv("NOTIONAL_USD", "0"))    # optional: if >0, router will size by notional if supported

# Poll cadence
SLEEP_SECONDS = int(os.getenv("SIGNAL_POLL_SECONDS", "20"))

# Optional portfolio segregation
PORTFOLIO_ID = os.getenv("PORTFOLIO_ID")  # if set, only execute signals with this portfolio_id (or NULL if ALL)

# Idempotency window (minutes)
DEDUPE_MINUTES = int(os.getenv("DEDUPE_MINUTES", "10"))

# ----------------
# DB
# ----------------

def get_conn():
    return psycopg2.connect(DB_URL)

def fetch_unprocessed_signals(conn):
    """
    Expected 'signals' table minimal columns:
      id (pk), symbol TEXT, side TEXT ('buy'/'sell'), strength NUMERIC, created_at timestamptz
    Optional (if used): portfolio_id TEXT
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        q = """
        SELECT id, symbol, side, strength, created_at, portfolio_id
        FROM signals
        WHERE processed_at IS NULL
          AND strength >= %s
        ORDER BY created_at ASC
        LIMIT 50;
        """
        cur.execute(q, (str(MIN_STRENGTH),))
        rows = cur.fetchall()
        return rows

def mark_signal(conn, signal_id, status, error=None, order_id=None, client_order_id=None):
    with conn.cursor() as cur:
        q = """
        UPDATE signals
        SET processed_at = NOW(),
            status = %s,
            error = %s,
            exec_order_id = %s,
            client_order_id = %s
        WHERE id = %s
        """
        cur.execute(q, (status, error, order_id, client_order_id, signal_id))
    conn.commit()

# ----------------
# Core
# ----------------

def acceptable(symbol: str, side: str, portfolio_id: str|None):
    if symbol.upper() not in SYMBOLS:
        return False, "symbol not allowed"
    if side.lower() not in ("buy", "sell"):
        return False, "invalid side"
    if PORTFOLIO_ID and portfolio_id and portfolio_id != PORTFOLIO_ID:
        return False, "portfolio mismatch"
    if PORTFOLIO_ID and portfolio_id is None:
        # enforce only my portfolio signals if env asks for it
        return False, "portfolio missing"
    return True, None

def size_for(symbol: str, side: str):
    # priority: NOTIONAL if set; else FIXED_QTY
    if NOTIONAL_USD > 0:
        return None, float(NOTIONAL_USD)  # (qty, notional)
    return float(FIXED_QTY), None

def process_one(tc: TradingClient, sig, conn):
    sid = sig["id"]
    symbol = sig["symbol"].upper()
    side = sig["side"].lower()
    strength = Decimal(str(sig["strength"]))
    portfolio_id = sig.get("portfolio_id")

    ok, why = acceptable(symbol, side, portfolio_id)
    if not ok:
        log.info("skip signal %s %s: %s", symbol, side, why)
        mark_signal(conn, sid, status="skipped", error=why)
        return

    qty, notional = size_for(symbol, side)

    # Build a unique-ish COID that the router can forward (router may override).
    coid = f"sigexec-{symbol}-{uuid.uuid4().hex[:6]}"

    # Route entry via order_router (handles AH safety, dedupe, spread, opposite cancel)
    try:
        o = place_entry(
            trading=tc,
            symbol=symbol,
            side=side,                         # 'buy' or 'sell'
            qty=qty,                           # may be None when using notional
            notional=notional,                 # may be None
            use_limit=False,                   # router will choose market/limit per rules
            allow_after_hours=True,            # router enforces AH rules (e.g., stops not AH)
            client_order_id=coid
        )
        log.info("submitted %s %s qty=%s notional=%s -> order_id=%s",
                 symbol, side, qty, notional, getattr(o, "id", None))
        mark_signal(conn, sid, status="submitted",
                    error=None, order_id=getattr(o, "id", None), client_order_id=coid)
    except Exception as e:
        err = str(e)
        log.warning("signal %s %s failed: %s", symbol, side, err)
        mark_signal(conn, sid, status="error", error=err)

def loop():
    key = os.getenv("ALPACA_API_KEY")
    sec = os.getenv("ALPACA_API_SECRET")
    paper = env_bool("ALPACA_PAPER", True)
    tc = TradingClient(key, sec, paper=paper)

    while True:
        try:
            with get_conn() as conn:
                sigs = fetch_unprocessed_signals(conn)
                if not sigs:
                    log.info("no new signals")
                for sig in sigs:
                    process_one(tc, sig, conn)
        except Exception as e:
            log.exception("main loop error: %s", e)
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    loop()


# tools/sync_orders.py
from __future__ import annotations
import os, sys, json, requests
from datetime import datetime, timezone, timedelta

# DB
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def log(msg, level="INFO"):
    levels = ["DEBUG","INFO","WARN","ERROR"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        print(f"{ts} {level} sync_orders | {msg}", flush=True)

def _conn():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg.connect(dsn, autocommit=True) if HAVE3 else _pg2(dsn)
    # discrete vars
    if HAVE3:
        return psycopg.connect(
            host=os.getenv("PGHOST","localhost"),
            user=os.getenv("PGUSER","postgres"),
            password=os.getenv("PGPASSWORD","postgres"),
            dbname=os.getenv("PGDATABASE","ai_prototype"),
            port=os.getenv("PGPORT","5432"),
            autocommit=True,
        )
    return _pg2(None)

def _pg2(dsn):
    if dsn:
        conn = psycopg2.connect(dsn); conn.autocommit = True; return conn
    conn = psycopg2.connect(
        host=os.getenv("PGHOST","localhost"),
        user=os.getenv("PGUSER","postgres"),
        password=os.getenv("PGPASSWORD","postgres"),
        dbname=os.getenv("PGDATABASE","ai_prototype"),
        port=os.getenv("PGPORT","5432"),
    )
    conn.autocommit = True
    return conn

def upsert_order(cur, o):
    sql = """
    INSERT INTO orders (id, client_order_id, symbol, side, order_type, order_class, qty, filled_qty, filled_avg_price,
                        status, time_in_force, limit_price, stop_price, extended_hours,
                        created_at, submitted_at, updated_at, filled_at, canceled_at, expires_at)
    VALUES (%(id)s, %(client_order_id)s, %(symbol)s, %(side)s, %(order_type)s, %(order_class)s, %(qty)s, %(filled_qty)s,
            %(filled_avg_price)s, %(status)s, %(time_in_force)s, %(limit_price)s, %(stop_price)s, %(extended_hours)s,
            %(created_at)s, %(submitted_at)s, %(updated_at)s, %(filled_at)s, %(canceled_at)s, %(expires_at)s)
    ON CONFLICT (id) DO UPDATE SET
      client_order_id=EXCLUDED.client_order_id,
      symbol=EXCLUDED.symbol,
      side=EXCLUDED.side,
      order_type=EXCLUDED.order_type,
      order_class=EXCLUDED.order_class,
      qty=EXCLUDED.qty,
      filled_qty=EXCLUDED.filled_qty,
      filled_avg_price=EXCLUDED.filled_avg_price,
      status=EXCLUDED.status,
      time_in_force=EXCLUDED.time_in_force,
      limit_price=EXCLUDED.limit_price,
      stop_price=EXCLUDED.stop_price,
      extended_hours=EXCLUDED.extended_hours,
      created_at=EXCLUDED.created_at,
      submitted_at=EXCLUDED.submitted_at,
      updated_at=EXCLUDED.updated_at,
      filled_at=EXCLUDED.filled_at,
      canceled_at=EXCLUDED.canceled_at,
      expires_at=EXCLUDED.expires_at;
    """
    def ts(x):
        if not x: return None
        return datetime.fromisoformat(str(x).replace("Z","+00:00"))
    params = {
        "id": o.get("id"),
        "client_order_id": o.get("client_order_id"),
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "order_type": (o.get("type") or o.get("order_type")),
        "order_class": o.get("order_class"),
        "qty": float(o["qty"]) if o.get("qty") is not None else None,
        "filled_qty": float(o["filled_qty"]) if o.get("filled_qty") is not None else None,
        "filled_avg_price": float(o["filled_avg_price"]) if o.get("filled_avg_price") else None,
        "status": o.get("status"),
        "time_in_force": o.get("time_in_force"),
        "limit_price": float(o["limit_price"]) if o.get("limit_price") else None,
        "stop_price": float(o["stop_price"]) if o.get("stop_price") else None,
        "extended_hours": bool(o.get("extended_hours")),
        "created_at": ts(o.get("created_at")),
        "submitted_at": ts(o.get("submitted_at")),
        "updated_at": ts(o.get("updated_at")),
        "filled_at": ts(o.get("filled_at")),
        "canceled_at": ts(o.get("canceled_at")),
        "expires_at": ts(o.get("expires_at")),
    }
    cur.execute(sql, params)

def fetch_orders(status="all", limit=200, after: datetime | None = None):
    h = {
        "APCA-API-KEY-ID": API_KEY or "",
        "APCA-API-SECRET-KEY": API_SECRET or "",
        "Content-Type": "application/json",
    }
    url = f"{ALPACA_BASE_URL}/v2/orders"
    params = {"status": status, "nested": "true", "limit": str(limit)}
    if after is not None:
        params["after"] = after.isoformat()
    r = requests.get(url, headers=h, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def main():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("ALPACA_API_KEY/SECRET missing in env")

    # Look back 7 days by default
    lookback_days = int(os.getenv("ORDERS_LOOKBACK_DAYS","7"))
    after = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    orders = fetch_orders(status="all", after=after)
    log(f"fetched {len(orders)} orders since {after.isoformat()}")

    conn = _conn()
    with conn.cursor() as cur:
        for o in orders:
            upsert_order(cur, o)
    conn.close()
    log("sync complete")

if __name__ == "__main__":
    main()

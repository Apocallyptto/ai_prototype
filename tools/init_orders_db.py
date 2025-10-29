# tools/init_orders_db.py
from __future__ import annotations
import os, sys, logging
import psycopg2

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO").upper(),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("init_orders_db")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")

DDL = """
CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,              -- Alpaca id
    client_order_id TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,                 -- buy/sell
    order_type      TEXT,                          -- limit/market/stop/...
    order_class     TEXT,                          -- bracket/oto/oco/...
    status          TEXT,
    qty             NUMERIC,
    filled_qty      NUMERIC,
    filled_avg_price NUMERIC,
    limit_price     NUMERIC,
    stop_price      NUMERIC,
    notional        NUMERIC,
    time_in_force   TEXT,
    extended_hours  BOOLEAN,
    submitted_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    filled_at       TIMESTAMPTZ,
    canceled_at     TIMESTAMPTZ,
    expired_at      TIMESTAMPTZ,
    position_intent TEXT,
    asset_id        TEXT,
    asset_class     TEXT,
    legs            JSONB,
    raw             JSONB
);
"""

def main() -> None:
    log.info("INIT_ORDERS_DB | connecting %s", DB_URL)
    with psycopg2.connect(DB_URL) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(DDL)
    log.info("INIT_ORDERS_DB | done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("INIT_ORDERS_DB failed: %s", e)
        sys.exit(1)

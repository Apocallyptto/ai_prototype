-- db/init.sql
BEGIN;

-- ---- signals (with idempotency fields) ----
CREATE TABLE IF NOT EXISTS signals (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol       TEXT        NOT NULL,
    side         TEXT        NOT NULL,
    strength     DOUBLE PRECISION NOT NULL,
    source       TEXT        NOT NULL DEFAULT 'unknown',
    portfolio_id INTEGER     NOT NULL DEFAULT 1
);

-- Add idempotency columns if missing (safe on existing DB)
ALTER TABLE signals ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS processed_status TEXT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS processed_note TEXT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS alpaca_order_id TEXT;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signals_created_at
    ON signals (created_at);

CREATE INDEX IF NOT EXISTS idx_signals_portfolio_created
    ON signals (portfolio_id, created_at);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_side_created
    ON signals (symbol, side, created_at);

-- Fast fetch for executor (unprocessed)
CREATE INDEX IF NOT EXISTS idx_signals_unprocessed
    ON signals (portfolio_id, created_at)
    WHERE processed_at IS NULL;

-- ---- daily_pnl (if your pnl_recorder uses it) ----
CREATE TABLE IF NOT EXISTS daily_pnl (
    id           BIGSERIAL PRIMARY KEY,
    day          DATE NOT NULL UNIQUE,
    equity       DOUBLE PRECISION,
    cash         DOUBLE PRECISION,
    buying_power DOUBLE PRECISION,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_daily_pnl_day ON daily_pnl(day);

-- ---- orders (for sync_orders; safe generic schema) ----
CREATE TABLE IF NOT EXISTS orders (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    alpaca_order_id  TEXT UNIQUE,
    client_order_id  TEXT,
    symbol           TEXT,
    side             TEXT,
    qty              DOUBLE PRECISION,
    filled_qty       DOUBLE PRECISION,
    status           TEXT,
    order_type       TEXT,
    time_in_force    TEXT,
    limit_price      DOUBLE PRECISION,
    stop_price       DOUBLE PRECISION,
    submitted_at     TIMESTAMPTZ,
    filled_at        TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ,
    raw_json         JSONB
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol_created ON orders(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_status_created ON orders(status, created_at);

COMMIT;

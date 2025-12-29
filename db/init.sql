-- db/init.sql
BEGIN;

CREATE TABLE IF NOT EXISTS signals (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol        TEXT        NOT NULL,
    side          TEXT        NOT NULL,
    strength      DOUBLE PRECISION NOT NULL,
    source        TEXT        NOT NULL DEFAULT 'unknown',
    portfolio_id  INTEGER     NOT NULL DEFAULT 1,

    -- stabilita / idempotencia:
    processed_at  TIMESTAMPTZ,
    processed_status TEXT,
    processed_note   TEXT,
    alpaca_order_id  TEXT
);

CREATE INDEX IF NOT EXISTS idx_signals_created_at
    ON signals (created_at);

CREATE INDEX IF NOT EXISTS idx_signals_pid_created
    ON signals (portfolio_id, created_at);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_side_created
    ON signals (symbol, side, created_at);

CREATE INDEX IF NOT EXISTS idx_signals_unprocessed
    ON signals (portfolio_id, processed_at)
    WHERE processed_at IS NULL;

-- pre daily risk guard (keď ENABLE_DAILY_RISK_GUARD=1)
CREATE TABLE IF NOT EXISTS daily_pnl (
    day           DATE PRIMARY KEY,
    pnl_usd       DOUBLE PRECISION NOT NULL DEFAULT 0,
    drawdown_pct  DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;

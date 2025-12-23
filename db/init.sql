-- db/init.sql
-- Minimal schema needed for ai_prototype (at least signals table).
-- Safe to run repeatedly.

BEGIN;

CREATE TABLE IF NOT EXISTS signals (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol       TEXT        NOT NULL,
    side         TEXT        NOT NULL,
    strength     DOUBLE PRECISION NOT NULL,
    source       TEXT        NOT NULL DEFAULT 'unknown',
    portfolio_id INTEGER     NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_signals_created_at
    ON signals (created_at);

CREATE INDEX IF NOT EXISTS idx_signals_portfolio_created
    ON signals (portfolio_id, created_at);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_side_created
    ON signals (symbol, side, created_at);

COMMIT;

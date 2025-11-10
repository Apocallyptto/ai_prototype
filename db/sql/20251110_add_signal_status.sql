-- db/sql/20251110_add_signal_status.sql

-- Add bookkeeping columns for the signal runner
ALTER TABLE signals
  ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS status_reason TEXT;

-- Backfill sensible defaults
UPDATE signals
SET status = COALESCE(status, 'pending')
WHERE status IS NULL;

-- Helpful indexes for the polling query
CREATE INDEX IF NOT EXISTS idx_signals_processed_at ON signals (processed_at);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);
CREATE INDEX IF NOT EXISTS idx_signals_strength_created_at ON signals (strength, created_at DESC);

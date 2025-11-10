-- Minimal, non-destructive migration for signals execution bookkeeping.
-- Adjust table/column names if your schema differs.

-- 1) Add columns if not present
ALTER TABLE IF EXISTS signals
  ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS status TEXT,             -- 'queued'|'submitted'|'skipped'|'error'
  ADD COLUMN IF NOT EXISTS error TEXT,              -- last error (optional)
  ADD COLUMN IF NOT EXISTS exec_order_id TEXT,      -- last created order id (optional)
  ADD COLUMN IF NOT EXISTS client_order_id TEXT;    -- client id we sent (optional)

-- 2) Helpful filtered index for polling
CREATE INDEX IF NOT EXISTS signals_unprocessed_idx
  ON signals (created_at)
  WHERE processed_at IS NULL;

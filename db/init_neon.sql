-- signals: add text symbol+timestamps+portfolio_id and helpful indexes
ALTER TABLE IF EXISTS public.signals
  ADD COLUMN IF NOT EXISTS symbol        TEXT,
  ADD COLUMN IF NOT EXISTS side          TEXT CHECK (side IN ('buy','sell')) DEFAULT 'buy',
  ADD COLUMN IF NOT EXISTS strength      DOUBLE PRECISION DEFAULT 0,
  ADD COLUMN IF NOT EXISTS ts            TIMESTAMPTZ DEFAULT now(),
  ADD COLUMN IF NOT EXISTS created_at    TIMESTAMPTZ DEFAULT now(),
  ADD COLUMN IF NOT EXISTS portfolio_id  INT DEFAULT 1;

-- backfill symbol from symbols(id -> ticker) if you had symbol_id
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='signals' AND column_name='symbol_id') THEN
    UPDATE public.signals s
      SET symbol = sym.ticker
    FROM public.symbols AS sym
    WHERE s.symbol_id IS NOT NULL AND s.symbol IS NULL AND sym.id = s.symbol_id;
  END IF;
END$$;

CREATE INDEX IF NOT EXISTS idx_signals_created_at
  ON public.signals (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_side_time
  ON public.signals (symbol, side, created_at DESC);

-- orders: id must be TEXT (Alpaca uses UUID-like ids) + full column set used by upsert
DO $$
DECLARE seq regclass;
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='orders' AND column_name='id' AND data_type IN ('bigint','integer','smallint')
  ) THEN
    SELECT pg_get_serial_sequence('public.orders','id') INTO seq;
    IF seq IS NOT NULL THEN
      EXECUTE 'ALTER TABLE public.orders ALTER COLUMN id DROP DEFAULT';
      EXECUTE format('ALTER SEQUENCE %s OWNED BY NONE', seq);
    END IF;
    EXECUTE 'ALTER TABLE public.orders DROP CONSTRAINT IF EXISTS orders_pkey';
    EXECUTE 'ALTER TABLE public.orders ALTER COLUMN id TYPE TEXT USING id::text';
    EXECUTE 'ALTER TABLE public.orders ADD PRIMARY KEY (id)';
  END IF;
END$$;

ALTER TABLE IF EXISTS public.orders
  ADD COLUMN IF NOT EXISTS client_order_id   TEXT,
  ADD COLUMN IF NOT EXISTS symbol            TEXT,
  ADD COLUMN IF NOT EXISTS side              TEXT,
  ADD COLUMN IF NOT EXISTS order_type        TEXT,
  ADD COLUMN IF NOT EXISTS order_class       TEXT,
  ADD COLUMN IF NOT EXISTS qty               TEXT,
  ADD COLUMN IF NOT EXISTS filled_qty        TEXT,
  ADD COLUMN IF NOT EXISTS status            TEXT,
  ADD COLUMN IF NOT EXISTS limit_price       TEXT,
  ADD COLUMN IF NOT EXISTS stop_price        TEXT,
  ADD COLUMN IF NOT EXISTS filled_avg_price  TEXT,
  ADD COLUMN IF NOT EXISTS time_in_force     TEXT,
  ADD COLUMN IF NOT EXISTS extended_hours    BOOLEAN,
  ADD COLUMN IF NOT EXISTS created_at        TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS updated_at        TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS submitted_at      TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS filled_at         TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS canceled_at       TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS expired_at        TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS failed_at         TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON public.orders (client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_time     ON public.orders (symbol, created_at DESC);

-- enforce public search_path for your Neon role (optional)
-- ALTER ROLE "neondb_owner" IN DATABASE "neondb" SET search_path = public;

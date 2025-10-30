# jobs/scale_strength.py
import os, logging, math
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scale_strength")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
HIST_LIMIT = int(os.getenv("STRENGTH_HIST_LIMIT", "100"))

SQL_SELECT = """
SELECT id, symbol, strength
FROM signals
WHERE created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC
"""

def main():
    log.info("=== Scaling signal strengths (per symbol z-score) ===")
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        # read recent signals
        df = pd.read_sql_query(text(SQL_SELECT), conn)
        if df.empty:
            log.info("no signals to scale")
            return

        # ensure column exists (idempotent)
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='signals' AND column_name='scaled_strength'
                ) THEN
                    ALTER TABLE signals ADD COLUMN scaled_strength DOUBLE PRECISION;
                END IF;
            END$$;
        """))

        # per-symbol z-score on last N rows, then normalize to [0,1]
        out_parts = []
        for sym, grp in df.groupby("symbol"):
            sub = grp.head(HIST_LIMIT)
            m = float(sub["strength"].mean())
            s = float(sub["strength"].std(ddof=0)) if len(sub) > 1 else 0.0
            if s == 0.0:
                # degenerate variance → set all to 0.5 midscale
                g = grp.copy()
                g["scaled_strength"] = 0.5
                out_parts.append(g)
                continue
            z = (grp["strength"] - m) / s
            z = z.clip(-3, 3)
            # normalize to [0,1] across this group's z-window
            zmin, zmax = float(z.min()), float(z.max())
            rng = (zmax - zmin) if (zmax - zmin) > 1e-9 else 1.0
            scaled = (z - zmin) / rng
            g = grp.copy()
            g["scaled_strength"] = scaled
            out_parts.append(g)

        out = pd.concat(out_parts, ignore_index=True)

        # bulk update
        # Use a small batch to avoid gigantic statements on very large tables
        upd = text("UPDATE signals SET scaled_strength=:val WHERE id=:id")
        batch = [{"val": float(v), "id": int(i)} for i, v in out[["id","scaled_strength"]].itertuples(index=False, name=None)]
        # chunked execution
        CHUNK = 2000
        for start in range(0, len(batch), CHUNK):
            conn.execute(upd, batch[start:start+CHUNK])

        conn.commit()
        log.info("Updated %d rows with scaled strengths", len(out))

if __name__ == "__main__":
    main()

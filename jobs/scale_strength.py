# jobs/scale_strength.py
import os, psycopg2, pandas as pd, logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scale_strength")

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
HIST_LIMIT = int(os.getenv("STRENGTH_HIST_LIMIT", "100"))


def main():
    log.info("=== Scaling signal strengths (per symbol z-score) ===")
    with psycopg2.connect(DB_URL) as conn:
        df = pd.read_sql("""
            SELECT id, symbol, strength
            FROM signals
            WHERE created_at >= NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
        """, conn)

    if df.empty:
        log.info("no signals to scale")
        return

    # compute per-symbol z-score scaling
    scaled_rows = []
    for sym, grp in df.groupby("symbol"):
        sub = grp.head(HIST_LIMIT)
        m = sub["strength"].mean()
        s = sub["strength"].std(ddof=0) or 1e-9
        z = (grp["strength"] - m) / s
        scaled = z.clip(-3, 3)
        # normalize to [0,1]
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-9)
        grp = grp.copy()
        grp["scaled_strength"] = scaled
        scaled_rows.append(grp)

    out = pd.concat(scaled_rows)
    with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
        # add column if not exists
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='signals' AND column_name='scaled_strength'
                ) THEN
                    ALTER TABLE signals ADD COLUMN scaled_strength DOUBLE PRECISION;
                END IF;
            END$$;
        """)
        for _, r in out.iterrows():
            cur.execute("UPDATE signals SET scaled_strength=%s WHERE id=%s",
                        (float(r["scaled_strength"]), int(r["id"])))
        conn.commit()

    log.info("Updated %d rows with scaled strengths", len(out))


if __name__ == "__main__":
    main()

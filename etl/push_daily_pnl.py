# Upserts a daily PnL row (per portfolio_id, date) so repeated runs are safe.

import os
from datetime import datetime, timezone, date
import sqlalchemy as sa

def engine_from_env():
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ.get('DB_PORT','5432')}/{os.environ['DB_NAME']}?sslmode=require"
    )
    return sa.create_engine(url, pool_pre_ping=True)

def upsert_daily_pnl(conn, row: dict):
    """
    row = {
      'portfolio_id': int,
      'date': datetime.date,
      'realized': float,
      'unrealized': float,
      'fees': float,
    }
    """
    stmt = sa.text("""
        INSERT INTO daily_pnl (portfolio_id, date, realized, unrealized, fees)
        VALUES (:portfolio_id, :date, :realized, :unrealized, :fees)
        ON CONFLICT (portfolio_id, date)
        DO UPDATE SET
            realized   = EXCLUDED.realized,
            unrealized = EXCLUDED.unrealized,
            fees       = EXCLUDED.fees
    """)
    conn.execute(stmt, row)

def main():
    eng = engine_from_env()

    # You can replace these with your real PnL numbers
    today_utc = datetime.now(timezone.utc).date()
    row = {
        "portfolio_id": 1,
        "date": today_utc,
        "realized": 0.0,
        "unrealized": 0.0,
        "fees": 0.0,
    }

    with eng.begin() as conn:
        upsert_daily_pnl(conn, row)

    print(f"✅ Upserted daily_pnl for portfolio_id={row['portfolio_id']} on {row['date']}")

if __name__ == "__main__":
    main()

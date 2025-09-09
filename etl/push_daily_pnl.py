# etl/push_daily_pnl.py
import os
import datetime as dt
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Date, Numeric
from sqlalchemy.dialects.postgresql import insert as pg_insert

def engine_from_env():
    url = (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )
    return create_engine(url, pool_pre_ping=True)

def compute_daily_row():
    # TODO: keep your existing logic that computes realized/unrealized/fees
    # For now, this matches the row you were inserting:
    return {
        "portfolio_id": 1,
        "date": dt.date.today(),
        "realized": 0,
        "unrealized": 0,
        "fees": 0,
    }

def upsert_daily_pnl(engine, row: dict):
    meta = MetaData()
    daily_pnl = Table(
        "daily_pnl",
        meta,
        Column("portfolio_id", Integer, primary_key=True),
        Column("date", Date, primary_key=True),
        Column("realized", Numeric),
        Column("unrealized", Numeric),
        Column("fees", Numeric),
    )
    stmt = pg_insert(daily_pnl).values(row)
    stmt = stmt.on_conflict_do_update(
        index_elements=["portfolio_id", "date"],
        set_={
            "realized": stmt.excluded.realized,
            "unrealized": stmt.excluded.unrealized,
            "fees": stmt.excluded.fees,
        },
    )
    with engine.begin() as conn:
        conn.execute(stmt)

if __name__ == "__main__":
    eng = engine_from_env()
    row = compute_daily_row()
    upsert_daily_pnl(eng, row)
    print("✅ upserted daily_pnl:", row)

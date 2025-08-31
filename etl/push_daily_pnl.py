import os, pandas as pd, sqlalchemy as sa
from sqlalchemy.engine import URL
from datetime import date

url = URL.create(
    "postgresql+psycopg2",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=int(os.environ.get("DB_PORT", 5432)),
    database=os.environ["DB_NAME"],
    query={"sslmode":"require","channel_binding":"require"},
)
engine = sa.create_engine(url, pool_pre_ping=True)

# TODO: replace with your real calculations
row = pd.DataFrame([{
    "portfolio_id": 1,
    "date": date.today(),
    "realized": 0,
    "unrealized": 0,
    "fees": 0,
}])

with engine.begin() as conn:
    row.to_sql("daily_pnl", conn, if_exists="append", index=False)

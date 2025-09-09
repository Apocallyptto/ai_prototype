#!/usr/bin/env python3
"""
etl/push_daily_pnl.py

Writes today's daily PnL into the `daily_pnl` table using an idempotent UPSERT.
Primary key: (portfolio_id, date)

Env vars required (same as your smoke test):
  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

Optional env vars:
  PORTFOLIO_ID      -> default: 1
  PNL_REALIZED      -> default: 0
  PNL_UNREALIZED    -> default: 0
  PNL_FEES          -> default: 0
  PNL_DATE          -> default: today's UTC date (YYYY-MM-DD)

If a row already exists for (portfolio_id, date), its values are updated.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Dict, Any, List

import pandas as pd
from sqlalchemy import create_engine, text


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"❌ Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return v


def make_engine():
    user = _require_env("DB_USER")
    pwd = _require_env("DB_PASSWORD")
    host = _require_env("DB_HOST")
    port = _require_env("DB_PORT")
    db   = _require_env("DB_NAME")

    # Use sslmode=require as in your smoke test
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=require"
    return create_engine(url, pool_pre_ping=True, future=True)


def _parse_date_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_daily_pnl_rows() -> pd.DataFrame:
    """
    Build a DataFrame with columns:
    [portfolio_id, date, realized, unrealized, fees]

    Replace or extend this with your actual PnL computation if needed.
    For now it uses env overrides or zeros, which matches your previous run.
    """
    portfolio_id = int(os.getenv("PORTFOLIO_ID", "1"))

    # date: allow override, else UTC today
    d = os.getenv("PNL_DATE")
    pnl_date = _parse_date_yyyy_mm_dd(d) if d else datetime.utcnow().date()

    realized = float(os.getenv("PNL_REALIZED", "0"))
    unrealized = float(os.getenv("PNL_UNREALIZED", "0"))
    fees = float(os.getenv("PNL_FEES", "0"))

    df = pd.DataFrame(
        [{
            "portfolio_id": portfolio_id,
            "date": pnl_date,
            "realized": realized,
            "unrealized": unrealized,
            "fees": fees,
        }]
    )

    # Ensure 'date' is a Python date (not Timestamp) for psycopg2
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def upsert_daily_pnl(engine, df: pd.DataFrame) -> int:
    """
    Perform an UPSERT into daily_pnl for all rows in df.
    Returns number of rows processed.
    """
    required_cols = {"portfolio_id", "date", "realized", "unrealized", "fees"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    records: List[Dict[str, Any]] = df.to_dict(orient="records")

    # Postgres UPSERT via text() with executemany
    stmt = text("""
        INSERT INTO daily_pnl (portfolio_id, date, realized, unrealized, fees)
        VALUES (:portfolio_id, :date, :realized, :unrealized, :fees)
        ON CONFLICT (portfolio_id, date) DO UPDATE
        SET realized   = EXCLUDED.realized,
            unrealized = EXCLUDED.unrealized,
            fees       = EXCLUDED.fees
    """)

    with engine.begin() as conn:
        conn.execute(stmt, records)

    return len(records)


def main() -> None:
    engine = make_engine()
    df = build_daily_pnl_rows()
    n = upsert_daily_pnl(engine, df)

    # Friendly log output for Actions
    row = df.iloc[0].to_dict()
    print(
        "✅ daily_pnl upserted",
        f"(portfolio_id={row['portfolio_id']}, date={row['date']}, "
        f"realized={row['realized']}, unrealized={row['unrealized']}, fees={row['fees']})"
    )
    print(f"Rows processed: {n}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ETL failed: {e}", file=sys.stderr)
        sys.exit(1)

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        print(f"❌ Missing required env var: {var}", file=sys.stderr)
        sys.exit(1)
    return val


def make_engine() -> Engine:
    """
    Build a SQLAlchemy engine from DB_* env vars.
    Exits with code 1 if any are missing.
    """
    host = _require_env("DB_HOST")
    port = _require_env("DB_PORT")
    name = _require_env("DB_NAME")
    user = _require_env("DB_USER")
    pwd = _require_env("DB_PASSWORD")

    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True, future=True)


def _parse_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"Invalid float for {env_name}: {raw!r}")


def _parse_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Invalid int for {env_name}: {raw!r}")


def _parse_date(env_name: str, default: date) -> date:
    raw = os.getenv(env_name)
    if not raw:
        return default
    try:
        # allow YYYY-MM-DD or full ISO
        return datetime.fromisoformat(raw).date()
    except Exception:
        raise SystemExit(f"Invalid date for {env_name}: {raw!r} (expected YYYY-MM-DD)")


def build_daily_pnl_rows() -> pd.DataFrame:
    """
    Build a one-row DataFrame from env overrides (or sensible defaults).
    PORTFOLIO_ID, PNL_DATE, PNL_REALIZED, PNL_UNREALIZED, PNL_FEES
    """
    row = {
        "portfolio_id": _parse_int("PORTFOLIO_ID", 1),
        "date": _parse_date("PNL_DATE", date.today()),
        "realized": _parse_float("PNL_REALIZED", 0.0),
        "unrealized": _parse_float("PNL_UNREALIZED", 0.0),
        "fees": _parse_float("PNL_FEES", 0.0),
    }
    return pd.DataFrame([row], columns=["portfolio_id", "date", "realized", "unrealized", "fees"])


def upsert_daily_pnl(engine: Engine, df: pd.DataFrame) -> int:
    """
    Dialect-aware UPSERT into daily_pnl on (portfolio_id, date).
    Returns the number of rows attempted (len(df)).
    """
    if df.empty:
        return 0

    meta = MetaData()
    # reflect if present; if not present this will still allow insert using the explicit columns
    tbl = Table("daily_pnl", meta, autoload_with=engine, extend_existing=True)

    dialect = engine.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(tbl).values(df.to_dict(orient="records"))
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
        return len(df)

    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        stmt = sqlite_insert(tbl).values(df.to_dict(orient="records"))
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
        return len(df)

    # Generic fallback: try raw SQL with ON CONFLICT (works on PG & modern SQLite)
    rows = df.to_dict(orient="records")
    sql = """
        INSERT INTO daily_pnl (portfolio_id, date, realized, unrealized, fees)
        VALUES (:portfolio_id, :date, :realized, :unrealized, :fees)
        ON CONFLICT (portfolio_id, date)
        DO UPDATE SET
          realized = excluded.realized,
          unrealized = excluded.unrealized,
          fees = excluded.fees;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)
    return len(df)


def main(argv: Iterable[str] | None = None) -> int:
    engine = make_engine()
    df = build_daily_pnl_rows()
    n = upsert_daily_pnl(engine, df)
    print(f"✔ upserted {n} daily_pnl row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from datetime import date

import pandas as pd
from sqlalchemy import text, Integer, Date as SA_Date, Float

from etl.push_daily_pnl import (
    build_daily_pnl_rows,
    upsert_daily_pnl,
    make_engine,
)


def _typed_select_all():
    """
    Make the SELECT typed so SQLite returns a real Python date.
    """
    return text("""
        SELECT portfolio_id, date, realized, unrealized, fees
        FROM daily_pnl
    """).columns(
        portfolio_id=Integer(),
        date=SA_Date(),
        realized=Float(),
        unrealized=Float(),
        fees=Float(),
    )


def test_upsert_inserts_then_updates(engine):
    # INSERT
    df1 = pd.DataFrame(
        [{
            "portfolio_id": 1,
            "date": date(2025, 9, 9),
            "realized": 10.0,
            "unrealized": 5.0,
            "fees": 1.0,
        }]
    )
    n1 = upsert_daily_pnl(engine, df1)
    assert n1 == 1

    with engine.connect() as conn:
        row = conn.execute(_typed_select_all()).mappings().one()

    assert row["portfolio_id"] == 1
    assert row["date"] == date(2025, 9, 9)
    assert row["realized"] == 10.0
    assert row["unrealized"] == 5.0
    assert row["fees"] == 1.0

    # UPSERT -> update values, still 1 row total
    df2 = pd.DataFrame(
        [{
            "portfolio_id": 1,
            "date": date(2025, 9, 9),
            "realized": 20.0,
            "unrealized": 7.5,
            "fees": 2.0,
        }]
    )
    n2 = upsert_daily_pnl(engine, df2)
    assert n2 == 1

    with engine.connect() as conn:
        cnt = conn.execute(text("SELECT COUNT(*) FROM daily_pnl")).scalar_one()
        assert cnt == 1
        row = conn.execute(_typed_select_all()).mappings().one()

    assert row["realized"] == 20.0
    assert row["unrealized"] == 7.5
    assert row["fees"] == 2.0


def test_upsert_multiple_rows(engine):
    df = pd.DataFrame(
        [
            {"portfolio_id": 1, "date": date(2025, 9, 10), "realized": 1.0, "unrealized": 2.0, "fees": 0.1},
            {"portfolio_id": 2, "date": date(2025, 9, 10), "realized": 3.0, "unrealized": 4.0, "fees": 0.2},
        ]
    )
    n = upsert_daily_pnl(engine, df)
    assert n == 2

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM daily_pnl")).scalar_one()
        assert count == 2


def test_build_daily_pnl_rows_env_overrides(monkeypatch):
    monkeypatch.setenv("PORTFOLIO_ID", "42")
    monkeypatch.setenv("PNL_DATE", "2025-09-09")
    monkeypatch.setenv("PNL_REALIZED", "123.45")
    monkeypatch.setenv("PNL_UNREALIZED", "-6.78")
    monkeypatch.setenv("PNL_FEES", "0.99")

    df = build_daily_pnl_rows()
    assert list(df.columns) == ["portfolio_id", "date", "realized", "unrealized", "fees"]

    row = df.iloc[0].to_dict()
    assert row["portfolio_id"] == 42
    assert row["date"].isoformat() == "2025-09-09"
    assert row["realized"] == 123.45
    assert row["unrealized"] == -6.78
    assert row["fees"] == 0.99


def test_make_engine_missing_env_exits(monkeypatch):
    for var in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)

    try:
        make_engine()
        assert False, "make_engine should sys.exit when env is missing"
    except SystemExit as e:
        assert e.code == 1

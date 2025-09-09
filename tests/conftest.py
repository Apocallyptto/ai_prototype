import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

# Make repo root importable (so `etl` resolves)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_schema(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS daily_pnl (
        portfolio_id INTEGER NOT NULL,
        date DATE NOT NULL,
        realized DOUBLE PRECISION NOT NULL,
        unrealized DOUBLE PRECISION NOT NULL,
        fees DOUBLE PRECISION NOT NULL,
        PRIMARY KEY (portfolio_id, date)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


@pytest.fixture(scope="session")
def engine():
    """
    Use TEST_DB_URL if set; otherwise use in-memory SQLite.
    (No Docker required.)
    """
    url = os.getenv("TEST_DB_URL") or "sqlite+pysqlite:///:memory:"
    eng = create_engine(url, future=True)
    _ensure_schema(eng)
    return eng


@pytest.fixture(autouse=True)
def _clean_daily_pnl(engine):
    cleanup_sql = "DELETE FROM daily_pnl" if engine.dialect.name == "sqlite" else "TRUNCATE TABLE daily_pnl"
    with engine.begin() as conn:
        conn.execute(text(cleanup_sql))
    yield

import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

# Ensure repo root is on sys.path so `etl` package resolves
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_DB_URL_ENV = "TEST_DB_URL"


def _ensure_schema(engine):
    # Minimal schema for daily_pnl used by the ETL
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
def db_url():
    """
    Returns a SQLAlchemy URL for a running Postgres.
    Priority:
      1) $TEST_DB_URL (e.g., postgresql+psycopg2://user:pass@host:port/dbname)
      2) Spin up ephemeral Postgres via testcontainers
    """
    env_url = os.getenv(TEST_DB_URL_ENV)
    if env_url:
        yield env_url
        return

    try:
        from testcontainers.postgres import PostgresContainer
    except Exception as e:
        pytest.skip(f"testcontainers not available and {TEST_DB_URL_ENV} not set: {e}")

    # Pin a common image
    with PostgresContainer("postgres:16-alpine") as pg:
        # testcontainers gives a SQLAlchemy-compatible URL
        url = pg.get_connection_url()
        yield url


@pytest.fixture(scope="session")
def engine(db_url):
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    _ensure_schema(engine)
    return engine


@pytest.fixture(autouse=True)
def _clean_daily_pnl(engine):
    # Truncate daily_pnl before each test for isolation
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE daily_pnl;"))
    yield
    # nothing to do post-test

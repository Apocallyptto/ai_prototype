import os
from functools import lru_cache

import sqlalchemy as sa

# Default: lokálny docker Postgres kontajner
DEFAULT_DB_URL = "postgresql+psycopg2://postgres:postgres@postgres:5432/trader"


def make_engine() -> sa.Engine:
    """
    Vytvorí SQLAlchemy engine.

    Priorita:
      1) DB_URL env var
      2) DEFAULT_DB_URL (lokálny postgres kontajner)
    """
    url = os.getenv("DB_URL") or DEFAULT_DB_URL
    engine = sa.create_engine(
        url,
        future=True,
        pool_pre_ping=True,
    )
    return engine


@lru_cache(maxsize=1)
def get_engine() -> sa.Engine:
    """
    Cached engine – používame napr. v utils, services atď.
    """
    return make_engine()

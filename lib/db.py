# lib/db.py
from __future__ import annotations

import os
import pathlib
import sqlalchemy as sa

# py3.11+ has tomllib in stdlib; fallback to tomli for earlier versions if needed
try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


def _load_streamlit_secrets() -> dict:
    """
    Load .streamlit/secrets.toml so CLI scripts (not Streamlit) can reuse the same creds.
    Returns {} if the file is missing.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    p = repo_root / ".streamlit" / "secrets.toml"
    if not p.exists():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)


def make_engine() -> sa.Engine:
    """
    Build a SQLAlchemy engine, preferring OS env vars, then falling back to .streamlit/secrets.toml
    Secrets format expected:

    [db]
    host     = "..."
    port     = "5432"
    dbname   = "neondb"
    user     = "neondb_owner"
    password = "..."
    sslmode  = "require"
    """
    secrets = _load_streamlit_secrets()
    dbs = secrets.get("db", {})

    host = os.getenv("DB_HOST", dbs.get("host"))
    port = int(os.getenv("DB_PORT", dbs.get("port", 5432)))
    name = os.getenv("DB_NAME", dbs.get("dbname"))
    user = os.getenv("DB_USER", dbs.get("user"))
    pwd  = os.getenv("DB_PASSWORD", dbs.get("password"))
    ssl  = os.getenv("DB_SSLMODE", dbs.get("sslmode", "require"))

    missing = [k for k, v in {
        "DB_HOST/host": host,
        "DB_NAME/dbname": name,
        "DB_USER/user": user,
        "DB_PASSWORD/password": pwd,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing DB settings: {', '.join(missing)}")

    url = sa.engine.URL.create(
        drivername="postgresql+psycopg2",
        username=user,
        password=pwd,
        host=host,
        port=port,
        database=name,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)

# lib/db.py
import os
import sqlalchemy as sa

def make_engine():
    # prefer env (CLI, Jobs, GitHub Actions)
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pw   = os.getenv("DB_PASSWORD")
    ssl  = os.getenv("DB_SSLMODE", "require")



    # optional: allow .streamlit/secrets.toml when running the UI locally
    try:
        import streamlit as st  # only available in UI
        if not host and "db" in st.secrets:
            s = st.secrets["db"]
            host = s.get("host", host)
            port = s.get("port", port)
            name = s.get("dbname", name)
            user = s.get("user", user)
            pw   = s.get("password", pw)
            ssl  = s.get("sslmode", ssl)
    except Exception:
        pass

    if not all([host, port, name, user, pw]):
        missing = [k for k, v in dict(
            DB_HOST=host, DB_PORT=port, DB_NAME=name, DB_USER=user, DB_PASSWORD=pw
        ).items() if not v]
        raise RuntimeError(f"Missing DB settings: {', '.join(missing)}")

    url = sa.engine.URL.create(
        "postgresql+psycopg2",
        username=user,
        password=pw,
        host=host,
        port=int(port),
        database=name,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)



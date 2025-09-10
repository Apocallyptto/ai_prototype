# pages/0_DB_Test.py
import os
import streamlit as st
import pandas as pd
import sqlalchemy as sa

st.set_page_config(page_title="DB Test", page_icon="🧪", layout="wide")
st.title("🧪 DB Test")

@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets.get("db", {})
    host = cfg.get("host") or os.getenv("DB_HOST")
    port = int(cfg.get("port") or os.getenv("DB_PORT", 5432))
    db   = cfg.get("dbname") or os.getenv("DB_NAME")
    user = cfg.get("user") or os.getenv("DB_USER")
    pw   = cfg.get("password") or os.getenv("DB_PASSWORD")
    ssl  = cfg.get("sslmode") or os.getenv("DB_SSLMODE", "require")
    url = sa.engine.URL.create(
        "postgresql+psycopg2", user, pw, host, port, db,
        query={"sslmode": ssl} if ssl else None,
    )
    return sa.create_engine(url, pool_pre_ping=True)

eng = get_engine()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Ping")
    try:
        with eng.connect() as c:
            n = c.execute(sa.text("select 1")).scalar_one()
        st.success(f"Connected (select 1 = {n})")
    except Exception as e:
        st.exception(e)

with col2:
    st.subheader("Tables")
    try:
        with eng.connect() as c:
            df = pd.read_sql(
                sa.text("""
                    select table_schema, table_name
                    from information_schema.tables
                    where table_schema not in ('pg_catalog','information_schema')
                    order by 1,2
                """), c
            )
        st.dataframe(df, use_container_width=True, height=300)
    except Exception as e:
        st.exception(e)

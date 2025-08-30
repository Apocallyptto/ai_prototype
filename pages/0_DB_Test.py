import streamlit as st, pandas as pd
from lib.db import get_engine

st.title("DB Connectivity Test")

# Show which keys are present (proves secrets loaded)
try:
    db_keys = sorted(st.secrets["db"].keys())
    st.write("Secrets keys:", db_keys)
except Exception as e:
    st.error("No [db] secrets loaded.")
    st.stop()

try:
    eng = get_engine()
    with eng.connect() as conn:
        st.success("Engine created, opening connection...")
        ver = conn.execute("select version()").scalar_one()
        st.code(ver)
        tables = pd.read_sql(
            "select table_name from information_schema.tables "
            "where table_schema='public' order by 1",
            conn,
        )
        st.dataframe(tables, use_container_width=True)
except Exception as e:
    st.error("Connection failed.")
    # Show non-secret-safe info
    st.write(type(e).__name__)
    # If SQLAlchemy wrapped a psycopg2 error, e.orig exists:
    if hasattr(e, "orig"):
        st.write("Driver error class:", type(e.orig).__name__)
        msg = str(e.orig)
        # Basic scrubbing in case the driver prints credentials
        for k in ("password",):
            msg = msg.replace(st.secrets['db']['password'], "***")
        st.code(msg)

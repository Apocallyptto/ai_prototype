import streamlit as st, pandas as pd
from sqlalchemy import text            # 👈 add this
from lib.db import get_engine

st.title("DB Connectivity Test")

# show loaded keys (optional)
st.write("Secrets keys:", sorted(st.secrets["db"].keys()))

try:
    with get_engine().connect() as conn:
        st.success("Engine created, opening connection...")

        ver = conn.execute(text("select version()")).scalar_one()   # 👈 wrap in text()
        st.code(ver)

        tables = pd.read_sql(
            text("select table_name from information_schema.tables where table_schema='public' order by 1"),
            conn,                                                   # 👈 keep same connection
        )
        st.dataframe(tables, use_container_width=True)

except Exception as e:
    st.error("Connection failed.")
    st.write(type(e).__name__)
    if hasattr(e, "orig"):
        st.write("Driver error class:", type(e.orig).__name__)

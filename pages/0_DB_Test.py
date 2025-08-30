# pages/0_DB_Test.py
import streamlit as st, pandas as pd
from lib.db import get_engine

st.title("DB Connectivity Test")

try:
    with get_engine().connect() as conn:
        v = conn.execute("select version()").scalar_one()
        st.success("Connected!")
        st.code(v)
        tables = pd.read_sql(
            "select table_name from information_schema.tables where table_schema='public' order by 1",
            conn,
        )
        st.dataframe(tables, use_container_width=True)
except Exception as e:
    st.error("Connection failed.")
    st.write(type(e).__name__)

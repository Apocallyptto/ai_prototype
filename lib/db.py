# lib/db.py
import streamlit as st
import sqlalchemy as sa
from sqlalchemy.engine import URL

@st.cache_resource
def get_engine():
    if "db" not in st.secrets:
        raise KeyError("db")

    s = st.secrets["db"]  # host, port, dbname, user, password

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=s["user"],
        password=s["password"],
        host=s["host"],                 # hostname only
        port=int(s.get("port", 5432)),
        database=s["dbname"],
        query={"sslmode": "require", "channel_binding": "require"},
    )
    return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=2)

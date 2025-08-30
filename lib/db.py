# lib/db.py
import streamlit as st
import sqlalchemy as sa
from sqlalchemy.engine import URL

@st.cache_resource
def get_engine():
    s = st.secrets["db"]          # requires a [db] block in Streamlit Cloud secrets
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=s["neondb_owner"],
        password=s["npg_XasVR6uImqh"],
        host=s["ep-raspy-smoke-ae8epwey-pooler.c-2.us-east-2.aws.neon.tech"],           # e.g. ep-...-pooler.c-2.us-east-2.aws.neon.tech
        port=int(s.get("port", 5432)),
        database=s["neondb"],     # e.g. neondb
        query={"sslmode": "require", "channel_binding": "require"},
    )
    return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=2)

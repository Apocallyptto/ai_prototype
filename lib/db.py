import streamlit as st
import sqlalchemy as sa

@st.cache_resource
def get_engine():
    s = st.secrets["db"]
    url = (
        f"postgresql+psycopg2://{s['user']}:{s['password']}"
        f"@{s['host']}:{s['port']}/{s['dbname']}?sslmode=require"
    )
    # If you ever get a “channel binding required” error, append:
    # url += "&channel_binding=require"
    return sa.create_engine(url, pool_pre_ping=True)

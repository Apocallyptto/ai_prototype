import streamlit as st, sqlalchemy as sa

@st.cache_resource
def get_engine():
    if "db" not in st.secrets:
        st.error("Database not configured. In Streamlit Cloud set Settings → Secrets with a [db] block.")
        st.stop()
    s = st.secrets["db"]
    url = (
        f"postgresql+psycopg2://{s['user']}:{s['password']}"
        f"@{s['host']}:{s['port']}/{s['dbname']}?sslmode=require"
    )
    # If you later see a channel-binding error, append:
    # url += "&channel_binding=require"
    return sa.create_engine(url, pool_pre_ping=True)

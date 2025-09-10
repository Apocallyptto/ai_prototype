import os
import pandas as pd
import sqlalchemy as sa
import streamlit as st

st.set_page_config(page_title="PnL Dashboard", layout="wide")

db_url = (
    f"postgresql+psycopg2://{os.environ.get('DB_USER')}:{os.environ.get('DB_PASSWORD')}"
    f"@{os.environ.get('DB_HOST')}:{os.environ.get('DB_PORT')}/{os.environ.get('DB_NAME')}?sslmode=require"
)

@st.cache_data(ttl=300)
def load_pnl():
    eng = sa.create_engine(db_url, pool_pre_ping=True)
    with eng.connect() as c:
        df = pd.read_sql(
            sa.text("select portfolio_id, date, realized, unrealized, fees from daily_pnl order by date desc, portfolio_id"),
            c,
            parse_dates=["date"],
        )
    df["net"] = df["realized"] + df["unrealized"] - df["fees"]
    return df

st.title("Daily PnL")

df = load_pnl()
if df.empty:
    st.info("No PnL yet.")
else:
    pid = st.selectbox("Portfolio", sorted(df.portfolio_id.unique()))
    dff = df[df.portfolio_id == pid].sort_values("date")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Net", f"{dff.net.iloc[-1]:,.2f}")
    c2.metric("Last Realized", f"{dff.realized.iloc[-1]:,.2f}")
    c3.metric("Last Unrealized", f"{dff.unrealized.iloc[-1]:,.2f}")
    c4.metric("Last Fees", f"{dff.fees.iloc[-1]:,.2f}")

    st.line_chart(dff.set_index("date")[["realized","unrealized","net"]])
    st.dataframe(dff.tail(50), use_container_width=True)

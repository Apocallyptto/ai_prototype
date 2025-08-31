import streamlit as st, pandas as pd
from sqlalchemy.dialects.postgresql import DATE, NUMERIC, INTEGER
from lib.db import get_engine

st.title("🛠 Admin: Import daily PnL CSV")
st.write("Columns required: portfolio_id,date,realized,unrealized,fees")

file = st.file_uploader("Upload CSV", type="csv")
if file:
    df = pd.read_csv(file)
    need = {"portfolio_id","date","realized","unrealized","fees"}
    if not need.issubset(map(str.lower, df.columns)):
        st.error(f"Missing columns. Need: {need}")
        st.stop()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    with get_engine().begin() as conn:
        df.to_sql(
            "daily_pnl", conn, if_exists="append", index=False,
            dtype={
                "portfolio_id": INTEGER(),
                "date": DATE(),
                "realized": NUMERIC(),
                "unrealized": NUMERIC(),
                "fees": NUMERIC(),
            }
        )
    st.success(f"Inserted {len(df)} rows.")

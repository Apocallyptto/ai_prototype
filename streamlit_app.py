# streamlit_app.py
import streamlit as st

st.set_page_config(page_title="AI Trading – Home", page_icon="📈", layout="wide")
st.title("AI Trading Dashboard")
st.write(
    "Use the sidebar to explore: DB Test, Signals, Orders, Risk, PnL, Charts, and the Dashboard."
)
st.info("Tip: store DB creds in `.streamlit/secrets.toml` so you don't export env vars every run.")

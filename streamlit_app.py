import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.title("📊 Trading Dashboard Prototype")

# Example equity curve
equity = pd.DataFrame({
    "Day": ["Day 1", "Day 2", "Day 3", "Day 4"],
    "Equity": [10000, 10250, 10100, 10500]
})

st.subheader("Equity Curve")
st.line_chart(equity.set_index("Day"))

# Example metrics
st.subheader("Key Metrics")
st.metric("Win Rate", "62%")
st.metric("Sharpe Ratio", "1.45")
st.metric("Total Return", "+5%")

# pages/2_Orders.py
from __future__ import annotations
import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import requests

# --- DB setup (psycopg3 preferred; fall back to psycopg2) ---
try:
    import psycopg  # type: ignore
    HAVE3 = True
except Exception:
    HAVE3 = False
    import psycopg2  # type: ignore

# --- Config (envs) ---
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/ai_prototype"
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

st.set_page_config(page_title="Orders", page_icon="📑", layout="wide")

# ---------- Helpers ----------
def _conn():
    if HAVE3:
        return psycopg.connect(DATABASE_URL)
    else:
        return psycopg2.connect(DATABASE_URL)

def _status_chip(s: str) -> str:
    s = (s or "").lower()
    color = {
        "new": "#6b7280",
        "accepted": "#2563eb",
        "pending_new": "#2563eb",
        "partially_filled": "#ea580c",
        "filled": "#16a34a",
        "canceled": "#ef4444",
        "expired": "#6b7280",
        "replaced": "#7c3aed",
        "stopped": "#ef4444",
        "suspended": "#ef4444",
        "rejected": "#ef4444",
    }.get(s, "#6b7280")
    return f"<span style='padding:2px 8px;border-radius:999px;background:{color};color:white;font-size:12px'>{s}</span>"

def _fmt_ts(x):
    if x is None:
        return ""
    try:
        return pd.to_datetime(x).tz_convert("Europe/Bratislava").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(x)

@st.cache_data(show_spinner=False, ttl=10)
def read_orders(symbols: Optional[List[str]], statuses: Optional[List[str]], days: int) -> pd.DataFrame:
    q = """
        SELECT id, client_order_id, symbol, side, order_type, order_class,
               qty, filled_qty, filled_avg_price, status, time_in_force,
               limit_price, stop_price, extended_hours,
               created_at, submitted_at, updated_at, filled_at, canceled_at, expires_at
        FROM orders
        WHERE created_at >= NOW() - INTERVAL %s
    """
    params: List[Any] = [f"{days} days"]
    if symbols and len(symbols) > 0:
        q += " AND symbol = ANY(%s)"
        params.append(symbols)
    if statuses and len(statuses) > 0:
        q += " AND status = ANY(%s)"
        params.append(statuses)
    q += " ORDER BY created_at DESC"

    with _conn() as c:
        with c.cursor() as cur:
            cur.execute(q, params)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    return df

def cancel_order(order_id: str) -> tuple[int, str]:
    if not API_KEY or not API_SECRET:
        return 400, "ALPACA_API_KEY/SECRET not set"
    h = {
        "APCA-API-KEY-ID": API_KEY,
        "APCA-API-SECRET-KEY": API_SECRET,
        "Content-Type": "application/json",
    }
    url = f"{ALPACA_BASE_URL}/v2/orders/{order_id}"
    r = requests.delete(url, headers=h, timeout=15)
    return r.status_code, r.text

# ---------- UI ----------
st.title("📑 Orders")

with st.sidebar:
    st.header("Filters")
    lookback_days = st.slider("Lookback (days)", 1, 30, 7)
    sym_input = st.text_input("Symbols (comma sep)", "AAPL,MSFT,SPY")
    symbols = [s.strip().upper() for s in sym_input.split(",") if s.strip()]
    statuses_all = ["new","accepted","pending_new","partially_filled","filled","canceled","expired","replaced","stopped","suspended","rejected"]
    status_sel = st.multiselect("Statuses", statuses_all, default=["new","accepted","pending_new","partially_filled"])
    auto = st.checkbox("Auto refresh", value=True)
    refresh_sec = st.slider("Refresh seconds", 5, 120, 15)
    st.markdown("---")
    st.caption("Set `ALPACA_API_KEY` and `ALPACA_API_SECRET` in your environment to enable Cancel.")

# Auto refresh (client-side)
if auto:
    st.experimental_set_query_params(_=int(time.time()))

df = read_orders(symbols, status_sel, lookback_days)

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", len(df))
k2.metric("Open", int((df["status"].str.lower().isin(["new","accepted","pending_new","partially_filled"])).sum()))
k3.metric("Filled", int((df["status"].str.lower()=="filled").sum()))
k4.metric("Canceled", int((df["status"].str.lower()=="canceled").sum()))
k5.metric("Symbols", df["symbol"].nunique() if not df.empty else 0)

if df.empty:
    st.info("No orders match your filters.")
    st.stop()

# Open orders (quick view + cancel)
st.subheader("Open Orders")
open_mask = df["status"].str.lower().isin(["new","accepted","pending_new","partially_filled"])
open_df = df.loc[open_mask].copy()

if open_df.empty:
    st.success("No open orders.")
else:
    slim = open_df[["id","symbol","side","order_type","order_class","qty","limit_price","stop_price","status","created_at","updated_at"]].copy()
    slim["created_at"] = slim["created_at"].apply(_fmt_ts)
    slim["updated_at"] = slim["updated_at"].apply(_fmt_ts)
    st.dataframe(slim, use_container_width=True, hide_index=True)

    with st.expander("Cancel orders"):
        ids_raw = st.text_area("Paste Alpaca Order IDs (one per line) to cancel", height=120, placeholder="order-id-1\norder-id-2")
        disabled = not bool(ids_raw.strip())
        if st.button("Cancel Selected", type="primary", disabled=disabled):
            if not API_KEY or not API_SECRET:
                st.error("Missing ALPACA_API_KEY/ALPACA_API_SECRET in environment.")
            else:
                ids = [x.strip() for x in ids_raw.splitlines() if x.strip()]
                results = []
                for oid in ids:
                    code, body = cancel_order(oid)
                    results.append({"id": oid, "status_code": code, "response": body[:500]})
                st.write(pd.DataFrame(results))
                st.cache_data.clear()  # bust cache so the table refreshes

# All orders (pretty)
st.subheader("All Orders")
show_cols = [
    "id","client_order_id","symbol","side","order_type","order_class","qty","filled_qty","filled_avg_price",
    "status","time_in_force","limit_price","stop_price","extended_hours",
    "created_at","submitted_at","updated_at","filled_at","canceled_at","expires_at"
]
dfv = df[show_cols].copy()

for c in ["created_at","submitted_at","updated_at","filled_at","canceled_at","expires_at"]:
    dfv[c] = dfv[c].apply(_fmt_ts)
dfv["status"] = dfv["status"].apply(lambda s: _status_chip(str(s)))
dfv["filled_avg_price"] = dfv["filled_avg_price"].fillna("").apply(lambda x: f"{x:.2f}" if x != "" else "")
dfv["limit_price"] = dfv["limit_price"].fillna("").apply(lambda x: f"{x:.2f}" if x != "" else "")
dfv["stop_price"] = dfv["stop_price"].fillna("").apply(lambda x: f"{x:.2f}" if x != "" else "")

st.write(dfv.to_html(escape=False, index=False), unsafe_allow_html=True)

st.caption("Tip: keep `jobs/cron_v2.py` running or run `python -m tools.sync_orders` periodically to keep this page fresh.")

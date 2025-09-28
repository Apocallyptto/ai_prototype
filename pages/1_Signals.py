import pandas as pd
import sqlalchemy as sa
import streamlit as st
from lib.db import make_engine

@st.cache_data(ttl=30)
def load_signals(sym_like: str, limit: int) -> pd.DataFrame:
    eng = make_engine()
    with eng.connect() as c:
        # works whether your schema has JSON `signal` or plain columns
        try:
            q = sa.text("""
                SELECT
                  s.ts,
                  COALESCE(sym.ticker, s.ticker)     AS ticker,
                  s.timeframe, s.model,
                  COALESCE((s.signal->>'side'), s.side) AS side,
                  COALESCE(NULLIF((s.signal->>'strength'), '')::float, s.strength) AS strength
                FROM signals s
                LEFT JOIN symbols sym ON sym.id = s.symbol_id
                WHERE (:sym = '' OR COALESCE(sym.ticker, s.ticker) ILIKE '%'||:sym||'%')
                ORDER BY s.ts DESC
                LIMIT :lim
            """)

            return pd.read_sql(q, c, params={"sym": sym_like, "lim": limit})
        except Exception:
            q2 = sa.text("""
                SELECT s.ts, sym.ticker, s.timeframe, s.model,
                       s.side, s.strength
                FROM signals s
                JOIN symbols sym ON sym.id = s.symbol_id
                WHERE (:sym = '' OR sym.ticker ILIKE '%' || :sym || '%')
                ORDER BY s.ts DESC
                LIMIT :lim
            """)
            return pd.read_sql(q2, c, params={"sym": sym_like, "lim": limit})

st.title("🔔 Signals")
sym_like = st.text_input("Filter by ticker (optional)", value="")
lim = st.slider("Max rows", 50, 1000, 200, 50)
df = load_signals(sym_like, lim)
st.dataframe(df, width="stretch")

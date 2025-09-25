import pandas as pd
import sqlalchemy as sa
import streamlit as st
from lib.db import make_engine

@st.cache_data(ttl=30)
def load_signals(sym_like: str, limit: int):
    eng = make_engine()
    with eng.connect() as c:
        # try JSON schema (signal jsonb) first
        try:
            q = sa.text("""
                SELECT s.ts, sym.ticker, s.timeframe, s.model,
                       s.signal->>'side' AS side,
                       (s.signal->>'strength')::float AS strength
                FROM signals s
                JOIN symbols sym ON sym.id = s.symbol_id
                WHERE (:sym = '' OR sym.ticker ILIKE '%%'||:sym||'%%')
                ORDER BY s.ts DESC
                LIMIT :lim
            """)
            return pd.read_sql(q, c, params={"sym": sym_like, "lim": limit})
        except Exception:
            # fallback: plain columns side/strength
            q2 = sa.text("""
                SELECT s.ts, sym.ticker, s.timeframe, s.model,
                       s.side, s.strength
                FROM signals s
                JOIN symbols sym ON sym.id = s.symbol_id
                WHERE (:sym = '' OR sym.ticker ILIKE '%%'||:sym||'%%')
                ORDER BY s.ts DESC
                LIMIT :lim
            """)
            return pd.read_sql(q2, c, params={"sym": sym_like, "lim": limit})

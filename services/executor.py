# services/executor.py (replace the function)
import pandas as pd
import sqlalchemy as sa

def latest_signals(conn, since_days: int = 5) -> pd.DataFrame:
    sql = sa.text("""
    with ranked as (
      select
        s.ts,
        COALESCE(sym.ticker, s.ticker) as symbol,
        COALESCE((s.signal->>'side'), s.side) as signal,
        COALESCE(NULLIF((s.signal->>'strength'), '')::float, s.strength) as strength,
        row_number() over (partition by COALESCE(sym.ticker, s.ticker) order by s.ts desc) as rn
      from signals s
      left join symbols sym on sym.id = s.symbol_id
      where s.ts >= now() - (:days || ' days')::interval
    )
    select ts, symbol, signal, strength
    from ranked
    where rn = 1
    order by symbol
    """)

    return pd.read_sql(sql, conn, params={"days": int(since_days)})

# services/executor.py (replace the function)
import pandas as pd
import sqlalchemy as sa

def latest_signals(conn, since_days: int = 5) -> pd.DataFrame:
    sql = sa.text("""
        with ranked as (
          select
            s.ts,
            sym.ticker as symbol,
            coalesce( (s.signal->>'side')::text, s.side ) as side,
            coalesce( (s.signal->>'strength')::float, s.strength ) as strength,
            row_number() over (partition by sym.ticker order by s.ts desc) as rn
          from signals s
          join symbols sym on sym.id = s.symbol_id
          where s.ts >= now() - (:days || ' days')::interval
        )
        select ts, symbol, side, strength
        from ranked
        where rn = 1
        order by symbol
    """)
    return pd.read_sql(sql, conn, params={"days": int(since_days)})

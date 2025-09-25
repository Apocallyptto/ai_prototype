q = sa.text("""
    SELECT
      s.ts,
      sym.ticker,
      s.timeframe,
      s.model,
      COALESCE(s.side,      s.signal->>'side')                      AS side,
      COALESCE(s.strength, (s.signal->>'strength')::float)          AS strength
    FROM signals s
    JOIN symbols sym ON sym.id = s.symbol_id
    WHERE (:sym = '' OR sym.ticker ILIKE '%' || :sym || '%')
    ORDER BY s.ts DESC
    LIMIT :lim
""")

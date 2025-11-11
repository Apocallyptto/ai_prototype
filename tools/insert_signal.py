# tools/insert_signal.py
import os, psycopg2

symbol       = os.getenv("SYMBOL", "AAPL")
side         = os.getenv("SIDE", "buy")               # buy | sell
strength     = float(os.getenv("STRENGTH", "0.95"))
portfolio_id = os.getenv("PORTFOLIO_ID", "paper-core")
source       = os.getenv("SOURCE", "rule")

db_url = os.environ["DB_URL"]
with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
    cur.execute("""
        INSERT INTO signals(symbol, side, strength, source, portfolio_id, status)
        SELECT %s,%s,%s,%s,%s,'pending'
        WHERE NOT EXISTS (
          SELECT 1 FROM signals
           WHERE processed_at IS NULL AND status='pending'
             AND symbol=%s AND side=%s
        );
    """, (symbol, side, strength, source, portfolio_id, symbol, side))
    conn.commit()
print(f"insert attempted: {symbol} {side} strength={strength} portfolio={portfolio_id}")

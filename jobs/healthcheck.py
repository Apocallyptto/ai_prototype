# jobs/healthcheck.py
from __future__ import annotations
import os, sys
import psycopg2, requests
from datetime import datetime, timezone

ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_API_SECRET", "")

def ok(msg): print(f"[OK ] {msg}")
def warn(msg): print(f"[WARN] {msg}")
def err(msg): print(f"[ERR] {msg}")

def _pg_conn():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=os.getenv("PGHOST","localhost"),
        user=os.getenv("PGUSER","postgres"),
        password=os.getenv("PGPASSWORD","postgres"),
        dbname=os.getenv("PGDATABASE","ai_prototype"),
        port=int(os.getenv("PGPORT","5432")),
    )

def check_db():
    try:
        with _pg_conn() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        ok("DB connect")
    except Exception as e:
        err(f"DB connect failed: {e}")
        return False

    needed = {
        "public.signals": ["symbol","side","strength","created_at","portfolio_id"],
        "public.orders":  ["id","client_order_id","symbol","side","order_class","created_at"],
    }
    try:
        with _pg_conn() as c, c.cursor() as cur:
            for tbl, cols in needed.items():
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema=%s AND table_name=%s
                """, (tbl.split(".")[0], tbl.split(".")[1]))
                present = {r[0] for r in cur.fetchall()}
                missing = [x for x in cols if x not in present]
                if missing:
                    warn(f"{tbl}: missing columns {missing}")
                else:
                    ok(f"{tbl}: columns present")
        return True
    except Exception as e:
        err(f"schema check failed: {e}")
        return False

def check_alpaca():
    try:
        s = requests.Session()
        s.headers.update({"APCA-API-KEY-ID":API_KEY,"APCA-API-SECRET-KEY":API_SECRET})
        r = s.get(f"{ALPACA_BASE_URL}/v2/account", timeout=10)
        if r.status_code == 200:
            js = r.json()
            ok(f"Alpaca account ok: {js.get('status')}, equity={js.get('equity')}")
            return True
        err(f"Alpaca account error: {r.status_code} {r.text}")
        return False
    except Exception as e:
        err(f"Alpaca ping failed: {e}")
        return False

def main():
    print(f"== Healthcheck {datetime.now(timezone.utc).isoformat()} ==")
    a = check_db()
    b = check_alpaca()
    if a and b: print("ALL GREEN")
    else: print("Issues found")

if __name__ == "__main__":
    sys.exit(main() or 0)

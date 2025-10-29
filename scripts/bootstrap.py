# scripts/bootstrap.py
from __future__ import annotations
import os
import subprocess
import time
import sys

def _run(cmd: list[str]) -> int:
    print("BOOT | run:", " ".join(cmd), flush=True)
    return subprocess.call(cmd)

def main():
    # Wait for Postgres to accept connections inside the compose network
    db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@postgres:5432/trader")
    print("BOOT | DB_URL =", db_url, flush=True)

    # Light wait loop so we don't crash if Postgres is still coming up
    for attempt in range(30):
        rc = _run([sys.executable, "-c", "import psycopg2, os; "
                 "import time; "
                 "import urllib.parse; "
                 f"import sys; "
                 f"dsn=os.getenv('DB_URL','{db_url}'); "
                 "import psycopg2; "
                 "try:\n"
                 "  conn=psycopg2.connect(dsn); conn.close(); sys.exit(0)\n"
                 "except Exception as e:\n"
                 "  print('BOOT | DB not ready:',e); sys.exit(1)"])
        if rc == 0:
            break
        time.sleep(2)

    # Initialize core schema
    _run([sys.executable, "-m", "tools.init_db"])
    # Ensure orders table exists for sync_orders
    _run([sys.executable, "-m", "tools.init_orders_db"])

    # Finally, start the main scheduler loop
    os.execv(sys.executable, [sys.executable, "-m", "jobs.cron_nn"])

if __name__ == "__main__":
    main()

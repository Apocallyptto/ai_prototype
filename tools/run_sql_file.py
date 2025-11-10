# tools/run_sql_file.py
import os, sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def main():
    if len(sys.argv) != 2:
        print("usage: python -m tools.run_sql_file <path/to/file.sql>")
        sys.exit(1)

    sql_path = sys.argv[1]
    if not os.path.exists(sql_path):
        print(f"file not found: {sql_path}")
        sys.exit(2)

    db_url = os.getenv("DB_URL")
    if not db_url:
        print("DB_URL env var is required")
        sys.exit(3)

    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()

    conn = psycopg2.connect(db_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        print(f"applied SQL: {sql_path}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

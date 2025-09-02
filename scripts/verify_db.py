# scripts/verify_db.py
import os
import sqlalchemy as sa
from sqlalchemy import text

url = (
    f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}?sslmode=require"
)
e = sa.create_engine(url)

with e.connect() as c:
    print(c.execute(text(
        "select column_name from information_schema.columns "
        "where table_name='orders' and column_name='filled_at'"
    )).fetchall())
    print(c.execute(text("select * from alembic_version")).fetchall())

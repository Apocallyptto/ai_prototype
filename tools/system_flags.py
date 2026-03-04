import os
from sqlalchemy import text
from tools.db import get_engine

def ensure_table():
    engine = get_engine()
    with engine.begin() as con:
        con.execute(text("""
        CREATE TABLE IF NOT EXISTS system_flags (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
        """))

def set_flag(key: str, value: str):
    ensure_table()
    engine = get_engine()
    with engine.begin() as con:
        con.execute(text("""
        INSERT INTO system_flags(key, value)
        VALUES (:k, :v)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW();
        """), {"k": key, "v": value})

def get_flag(key: str, default: str = "") -> str:
    ensure_table()
    engine = get_engine()
    with engine.connect() as con:
        r = con.execute(text("SELECT value FROM system_flags WHERE key=:k"), {"k": key}).fetchone()
        return (r[0] if r and r[0] is not None else default)
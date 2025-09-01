from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool
import os

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def _url_from_env():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASSWORD")
    if not all([host, name, user, pwd]):
        raise RuntimeError(
            "DB env vars missing. Set DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD or DATABASE_URL."
        )
    return (
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{name}"
        "?sslmode=require&channel_binding=require"
    )

def run_migrations_offline():
    context.configure(url=_url_from_env(), target_metadata=None, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": _url_from_env()},  # <<< DO NOT read from alembic.ini
        prefix="",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

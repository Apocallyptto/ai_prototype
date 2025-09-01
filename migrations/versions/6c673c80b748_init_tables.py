from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as psql

# revision identifiers, used by Alembic.
revision = "6c673c80b748"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "symbols",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.Text, nullable=False, unique=True),
    )

    op.create_table(
        "daily_pnl",
        sa.Column("portfolio_id", sa.Integer, nullable=False, server_default=sa.text("1")),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("realized", sa.Numeric, server_default=sa.text("0")),
        sa.Column("unrealized", sa.Numeric, server_default=sa.text("0")),
        sa.Column("fees", sa.Numeric, server_default=sa.text("0")),
        sa.PrimaryKeyConstraint("portfolio_id", "date"),
    )

    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("symbol_id", sa.Integer, sa.ForeignKey("symbols.id")),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("timeframe", sa.Text, server_default=sa.text("'1d'"), nullable=False),
        sa.Column("model", sa.Text, server_default=sa.text("'baseline'"), nullable=False),
        sa.Column("signal", psql.JSONB, nullable=False),
    )

    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("ts", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("symbol_id", sa.Integer, sa.ForeignKey("symbols.id")),
        sa.Column("side", sa.Text, nullable=False),
        sa.Column("qty", sa.Numeric, nullable=False),
        sa.Column("type", sa.Text, server_default=sa.text("'market'")),
        sa.Column("limit_price", sa.Numeric),
        sa.Column("status", sa.Text, server_default=sa.text("'new'")),
        sa.Column("meta", psql.JSONB, server_default=sa.text("'{}'::jsonb")),
    )

    op.create_index("idx_signals_symbol_ts", "signals", ["symbol_id", "ts"])
    op.create_index("idx_orders_symbol_ts", "orders", ["symbol_id", "ts"])
    op.create_index("idx_pnl_portfolio_date", "daily_pnl", ["portfolio_id", "date"])

def downgrade():
    op.drop_index("idx_pnl_portfolio_date", table_name="daily_pnl")
    op.drop_index("idx_orders_symbol_ts", table_name="orders")
    op.drop_index("idx_signals_symbol_ts", table_name="signals")
    op.drop_table("orders")
    op.drop_table("signals")
    op.drop_table("daily_pnl")
    op.drop_table("symbols")

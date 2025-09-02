"""add filled_at to orders

Revision ID: 01e34ac8170f
Revises: 6c673c80b748
Create Date: 2025-09-02 23:43:56.246868
"""

from alembic import op
import sqlalchemy as sa

revision = "<the new rev id>"
down_revision = "6c673c80b748"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column(
        "orders",
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True)
    )
    # optional: op.create_index("idx_orders_filled_at", "orders", ["filled_at"])

def downgrade():
    # optional: op.drop_index("idx_orders_filled_at", table_name="orders")
    op.drop_column("orders", "filled_at")

"""Disk usage daily snapshots per tenant and host.

Revision ID: 015
Revises: 014
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "disk_usage_daily" in tables:
        return
    op.create_table(
        "disk_usage_daily",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "snapshot_date", name="uq_disk_usage_daily"),
    )
    op.create_index(
        "ix_disk_usage_daily_tenant_date",
        "disk_usage_daily",
        ["tenant_id", "snapshot_date"],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "disk_usage_daily" not in tables:
        return
    op.drop_index("ix_disk_usage_daily_tenant_date", table_name="disk_usage_daily")
    op.drop_table("disk_usage_daily")

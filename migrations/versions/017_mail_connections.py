"""Per-tenant reusable OAuth mail connections.

Revision ID: 017
Revises: 016
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "mail_connections" in inspector.get_table_names():
        return
    op.create_table(
        "mail_connections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("mailbox_email", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("config_enc", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_mail_connections_tenant_id", "mail_connections", ["tenant_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "mail_connections" not in inspector.get_table_names():
        return
    op.drop_index("ix_mail_connections_tenant_id", table_name="mail_connections")
    op.drop_table("mail_connections")

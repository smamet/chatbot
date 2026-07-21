"""Pending reply HTML draft + edit audit log.

Revision ID: 011
Revises: 010
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    pending_cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "draft_html" not in pending_cols:
        op.add_column(
            "pending_replies",
            sa.Column("draft_html", sa.Text(), nullable=True),
        )

    tables = set(inspector.get_table_names())
    if "pending_reply_edits" not in tables:
        op.create_table(
            "pending_reply_edits",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("tenant_id", sa.Integer(), nullable=False),
            sa.Column("pending_reply_id", sa.Integer(), nullable=False),
            sa.Column("edited_by", sa.String(length=256), nullable=False, server_default=""),
            sa.Column("body_before", sa.Text(), nullable=False),
            sa.Column("body_after", sa.Text(), nullable=False),
            sa.Column("diff", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["pending_reply_id"], ["pending_replies.id"]),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_pending_reply_edits_tenant_reply",
            "pending_reply_edits",
            ["tenant_id", "pending_reply_id"],
        )


def downgrade() -> None:
    op.drop_index("ix_pending_reply_edits_tenant_reply", table_name="pending_reply_edits")
    op.drop_table("pending_reply_edits")
    op.drop_column("pending_replies", "draft_html")

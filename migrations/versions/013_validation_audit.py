"""Validation audit events + resolved_by on pending replies; rename client_user role.

Revision ID: 013
Revises: 012
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    pending_cols = {c["name"] for c in inspector.get_columns("pending_replies")}
    if "resolved_by" not in pending_cols:
        op.add_column(
            "pending_replies",
            sa.Column("resolved_by", sa.String(length=256), nullable=True),
        )
    if "resolved_at" not in pending_cols:
        op.add_column(
            "pending_replies",
            sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        )

    tables = set(inspector.get_table_names())
    if "pending_reply_audit_events" not in tables:
        op.create_table(
            "pending_reply_audit_events",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("tenant_id", sa.Integer(), nullable=False),
            sa.Column("pending_reply_id", sa.Integer(), nullable=False),
            sa.Column("action", sa.String(length=32), nullable=False),
            sa.Column("actor_email", sa.String(length=256), nullable=False, server_default=""),
            sa.Column("detail_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["pending_reply_id"], ["pending_replies.id"]),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_pending_reply_audit_tenant_reply",
            "pending_reply_audit_events",
            ["tenant_id", "pending_reply_id"],
        )
        op.create_index(
            "ix_pending_reply_audit_tenant_created",
            "pending_reply_audit_events",
            ["tenant_id", "created_at"],
        )

    op.execute(
        sa.text("UPDATE users SET role = 'client_operator' WHERE role = 'client_user'")
    )


def downgrade() -> None:
    op.execute(
        sa.text("UPDATE users SET role = 'client_user' WHERE role = 'client_operator'")
    )
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = set(inspector.get_table_names())
    if "pending_reply_audit_events" in tables:
        op.drop_index(
            "ix_pending_reply_audit_tenant_created",
            table_name="pending_reply_audit_events",
        )
        op.drop_index(
            "ix_pending_reply_audit_tenant_reply",
            table_name="pending_reply_audit_events",
        )
        op.drop_table("pending_reply_audit_events")

    pending_cols = {c["name"] for c in inspector.get_columns("pending_replies")}
    if "resolved_at" in pending_cols:
        op.drop_column("pending_replies", "resolved_at")
    if "resolved_by" in pending_cols:
        op.drop_column("pending_replies", "resolved_by")

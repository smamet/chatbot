"""Email threads, outbound message IDs, mail draft threading fields.

Revision ID: 019
Revises: 018
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SESSION_ID_TABLES = (
    "messages",
    "hook_events",
    "orders",
    "order_events",
    "pending_replies",
    "test_chat_sessions",
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    op.create_table(
        "email_threads",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("from_addr", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("thread_key", sa.String(length=16), nullable=False, server_default=""),
        sa.Column("root_message_id", sa.String(length=255), nullable=True),
        sa.Column("normalized_subject", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "tenant_id",
            "from_addr",
            "thread_key",
            name="uq_email_threads_tenant_from_key",
        ),
    )
    op.create_index("ix_email_threads_tenant_id", "email_threads", ["tenant_id"])
    op.create_index("ix_email_threads_tenant_from", "email_threads", ["tenant_id", "from_addr"])

    op.create_table(
        "outbound_email_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.String(length=255), nullable=False),
        sa.Column("in_reply_to", sa.String(length=255), nullable=True),
        sa.Column("references_header", sa.Text(), nullable=True),
        sa.Column("pending_reply_id", sa.Integer(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.ForeignKeyConstraint(["thread_id"], ["email_threads.id"]),
        sa.ForeignKeyConstraint(["pending_reply_id"], ["pending_replies.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "message_id", name="uq_outbound_email_tenant_message"),
    )
    op.create_index(
        "ix_outbound_email_messages_tenant_thread",
        "outbound_email_messages",
        ["tenant_id", "thread_id"],
    )

    mail_cols = {col["name"] for col in inspector.get_columns("mail_drafts")}
    if "thread_id" not in mail_cols:
        op.add_column("mail_drafts", sa.Column("thread_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_mail_drafts_thread_id",
            "mail_drafts",
            "email_threads",
            ["thread_id"],
            ["id"],
        )
    if "message_id" not in mail_cols:
        op.add_column(
            "mail_drafts",
            sa.Column("message_id", sa.String(length=255), nullable=False, server_default=""),
        )
    if "in_reply_to" not in mail_cols:
        op.add_column(
            "mail_drafts",
            sa.Column("in_reply_to", sa.String(length=255), nullable=False, server_default=""),
        )
    if "references_header" not in mail_cols:
        op.add_column("mail_drafts", sa.Column("references_header", sa.Text(), nullable=True))
    if "normalized_subject" not in mail_cols:
        op.add_column(
            "mail_drafts",
            sa.Column("normalized_subject", sa.String(length=512), nullable=False, server_default=""),
        )
    if "body_new" not in mail_cols:
        op.add_column("mail_drafts", sa.Column("body_new", sa.Text(), nullable=False, server_default=""))

    pending_cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "mail_draft_id" not in pending_cols:
        op.add_column("pending_replies", sa.Column("mail_draft_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_pending_replies_mail_draft_id",
            "pending_replies",
            "mail_drafts",
            ["mail_draft_id"],
            ["id"],
        )
    if "thread_id" not in pending_cols:
        op.add_column("pending_replies", sa.Column("thread_id", sa.Integer(), nullable=True))
        op.create_foreign_key(
            "fk_pending_replies_thread_id",
            "pending_replies",
            "email_threads",
            ["thread_id"],
            ["id"],
        )

    for table in _SESSION_ID_TABLES:
        cols = {col["name"] for col in inspector.get_columns(table)}
        if "session_id" in cols:
            op.alter_column(
                table,
                "session_id",
                existing_type=sa.String(length=256),
                type_=sa.String(length=512),
                existing_nullable=False if table != "messages" else True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    pending_cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "thread_id" in pending_cols:
        op.drop_constraint("fk_pending_replies_thread_id", "pending_replies", type_="foreignkey")
        op.drop_column("pending_replies", "thread_id")
    if "mail_draft_id" in pending_cols:
        op.drop_constraint("fk_pending_replies_mail_draft_id", "pending_replies", type_="foreignkey")
        op.drop_column("pending_replies", "mail_draft_id")

    mail_cols = {col["name"] for col in inspector.get_columns("mail_drafts")}
    for col in ("body_new", "normalized_subject", "references_header", "in_reply_to", "message_id"):
        if col in mail_cols:
            op.drop_column("mail_drafts", col)
    if "thread_id" in mail_cols:
        op.drop_constraint("fk_mail_drafts_thread_id", "mail_drafts", type_="foreignkey")
        op.drop_column("mail_drafts", "thread_id")

    op.drop_index("ix_outbound_email_messages_tenant_thread", table_name="outbound_email_messages")
    op.drop_table("outbound_email_messages")
    op.drop_index("ix_email_threads_tenant_from", table_name="email_threads")
    op.drop_index("ix_email_threads_tenant_id", table_name="email_threads")
    op.drop_table("email_threads")

    for table in _SESSION_ID_TABLES:
        cols = {col["name"] for col in inspector.get_columns(table)}
        if "session_id" in cols:
            op.alter_column(
                table,
                "session_id",
                existing_type=sa.String(length=512),
                type_=sa.String(length=256),
                existing_nullable=False if table != "messages" else True,
            )

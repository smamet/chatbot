"""IMAP UID ledger for skipped/processed mail without mail_drafts.

Revision ID: 007
Revises: 006
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mail_imap_uids",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=False),
        sa.Column("imap_uid", sa.String(length=128), nullable=False),
        sa.Column("disposition", sa.String(length=32), nullable=False, server_default="skipped"),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_mail_imap_uids_tenant_id", "mail_imap_uids", ["tenant_id"])
    op.create_index(
        "uq_mail_imap_uids_tenant_imap_uid",
        "mail_imap_uids",
        ["tenant_id", "imap_uid"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_mail_imap_uids_tenant_imap_uid", table_name="mail_imap_uids")
    op.drop_index("ix_mail_imap_uids_tenant_id", table_name="mail_imap_uids")
    op.drop_table("mail_imap_uids")

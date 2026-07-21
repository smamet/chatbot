"""Unique mail draft per tenant IMAP UID.

Revision ID: 006
Revises: 005
"""

from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_mail_drafts_tenant_imap_uid",
        "mail_drafts",
        ["tenant_id", "imap_uid"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_mail_drafts_tenant_imap_uid", table_name="mail_drafts")

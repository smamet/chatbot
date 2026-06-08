"""Pending reply quote fields.

Revision ID: 005
Revises: 004
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("pending_replies", sa.Column("hook_event_id", sa.Integer(), nullable=True))
    op.add_column(
        "pending_replies",
        sa.Column("fulfillment_kind", sa.String(length=32), nullable=False, server_default="reply_only"),
    )
    op.add_column("pending_replies", sa.Column("quote_proposal_json", sa.Text(), nullable=True))
    op.add_column("pending_replies", sa.Column("quote_resolved_json", sa.Text(), nullable=True))
    op.add_column("pending_replies", sa.Column("quote_external_id", sa.String(length=128), nullable=True))
    op.add_column("pending_replies", sa.Column("attachments_json", sa.Text(), nullable=True))
    op.add_column("pending_replies", sa.Column("fulfillment_error", sa.Text(), nullable=True))
    op.create_foreign_key(
        "fk_pending_replies_hook_event_id",
        "pending_replies",
        "hook_events",
        ["hook_event_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_pending_replies_hook_event_id", "pending_replies", type_="foreignkey")
    op.drop_column("pending_replies", "fulfillment_error")
    op.drop_column("pending_replies", "attachments_json")
    op.drop_column("pending_replies", "quote_external_id")
    op.drop_column("pending_replies", "quote_resolved_json")
    op.drop_column("pending_replies", "quote_proposal_json")
    op.drop_column("pending_replies", "fulfillment_kind")
    op.drop_column("pending_replies", "hook_event_id")

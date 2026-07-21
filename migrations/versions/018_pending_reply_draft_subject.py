"""Per-reply outbound email subject on pending_replies.

Revision ID: 018
Revises: 017
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    pending_cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "draft_subject" not in pending_cols:
        op.add_column(
            "pending_replies",
            sa.Column("draft_subject", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    pending_cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "draft_subject" in pending_cols:
        op.drop_column("pending_replies", "draft_subject")

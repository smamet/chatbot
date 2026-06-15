"""Pending reply ERPNext quotation modified snapshot.

Revision ID: 012
Revises: 011
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("pending_replies")}
    if "quote_erp_modified" not in columns:
        op.add_column(
            "pending_replies",
            sa.Column("quote_erp_modified", sa.String(length=64), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("pending_replies")}
    if "quote_erp_modified" in columns:
        op.drop_column("pending_replies", "quote_erp_modified")

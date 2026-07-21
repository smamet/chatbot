"""Last quote name on test chat sessions for PDF resume.

Revision ID: 009
Revises: 008
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "test_chat_sessions",
        sa.Column("last_quote_name", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("test_chat_sessions", "last_quote_name")

"""RAG context debug JSON on assistant messages.

Revision ID: 010
Revises: 009
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {col["name"] for col in inspect(bind).get_columns("messages")}
    if "context_debug_json" not in columns:
        op.add_column(
            "messages",
            sa.Column("context_debug_json", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("messages", "context_debug_json")

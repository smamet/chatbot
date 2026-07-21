"""Add created_by_user_id to test_chat_sessions.

Revision ID: 024
Revises: 023
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "024"
down_revision: Union[str, None] = "023"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "test_chat_sessions",
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_test_chat_sessions_created_by_user_id",
        "test_chat_sessions",
        "users",
        ["created_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_test_chat_sessions_created_by_user_id",
        "test_chat_sessions",
        ["created_by_user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_test_chat_sessions_created_by_user_id", table_name="test_chat_sessions")
    op.drop_constraint(
        "fk_test_chat_sessions_created_by_user_id",
        "test_chat_sessions",
        type_="foreignkey",
    )
    op.drop_column("test_chat_sessions", "created_by_user_id")

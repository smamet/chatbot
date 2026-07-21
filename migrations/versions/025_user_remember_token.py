"""Add remember_token_hash to users for persistent login.

Revision ID: 025
Revises: 024
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "025"
down_revision: Union[str, None] = "024"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("remember_token_hash", sa.String(length=256), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "remember_token_hash")

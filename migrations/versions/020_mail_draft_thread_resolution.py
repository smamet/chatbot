"""Mail draft thread resolution audit JSON.

Revision ID: 020
Revises: 019
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {col["name"] for col in inspector.get_columns("mail_drafts")}
    if "thread_resolution_json" not in cols:
        op.add_column("mail_drafts", sa.Column("thread_resolution_json", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {col["name"] for col in inspector.get_columns("mail_drafts")}
    if "thread_resolution_json" in cols:
        op.drop_column("mail_drafts", "thread_resolution_json")

"""Backfill mail_drafts.body_new from body_in using sanitize + reply parser.

Revision ID: 021
Revises: 020
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from chatbot.application.email_body_sanitize import prepare_email_body_new

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "mail_drafts" not in inspector.get_table_names():
        return
    cols = {col["name"] for col in inspector.get_columns("mail_drafts")}
    if "body_in" not in cols or "body_new" not in cols:
        return

    rows = bind.execute(sa.text("SELECT id, body_in, body_new FROM mail_drafts")).fetchall()
    for row in rows:
        sanitized = prepare_email_body_new(row.body_in or "")
        if sanitized == (row.body_new or ""):
            continue
        bind.execute(
            sa.text("UPDATE mail_drafts SET body_new = :body_new WHERE id = :id"),
            {"body_new": sanitized, "id": row.id},
        )


def downgrade() -> None:
    # One-way data backfill; previous body_new values are not recoverable.
    pass

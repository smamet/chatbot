"""One-time normalize pending_replies.draft_html for validation editor signatures.

Revision ID: 023
Revises: 022
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

from chatbot.adapters.mail.body_format import (
    email_draft_html_from_markdown,
    prepare_email_draft_html_for_editor,
)

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "pending_replies" not in inspector.get_table_names():
        return
    cols = {col["name"] for col in inspector.get_columns("pending_replies")}
    if "draft_html" not in cols or "draft_text" not in cols:
        return

    rows = bind.execute(
        sa.text(
            "SELECT id, channel, draft_html, draft_text FROM pending_replies"
        )
    ).fetchall()

    for row in rows:
        channel = (row.channel or "").lower()
        if channel != "email":
            continue

        current_html = row.draft_html or ""
        if current_html.strip():
            normalized = prepare_email_draft_html_for_editor(current_html)
        else:
            draft_text = (row.draft_text or "").strip()
            if not draft_text:
                continue
            normalized = email_draft_html_from_markdown(draft_text)

        if normalized == current_html:
            continue

        bind.execute(
            sa.text("UPDATE pending_replies SET draft_html = :draft_html WHERE id = :id"),
            {"draft_html": normalized, "id": row.id},
        )


def downgrade() -> None:
    pass

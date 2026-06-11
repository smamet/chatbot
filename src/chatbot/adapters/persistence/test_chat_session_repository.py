from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.orm import TestChatSessionRow
from chatbot.application.customer_access_gate import parse_session_identity


def test_chat_session_label(session_id: str) -> str:
    email, phone = parse_session_identity(session_id)
    if email:
        return email
    if phone:
        return phone
    if session_id.startswith("dashboard:"):
        return "Anonyme"
    return session_id


class TestChatSessionRepository:
    def __init__(self, session: Session, tenant_id: int) -> None:
        self._session = session
        self._tenant_id = tenant_id

    def upsert(
        self,
        session_id: str,
        *,
        label: str | None = None,
        last_quote_name: str | None = None,
    ) -> None:
        resolved_label = (label or test_chat_session_label(session_id)).strip() or "Anonyme"
        row = self._session.scalar(
            select(TestChatSessionRow).where(
                TestChatSessionRow.tenant_id == self._tenant_id,
                TestChatSessionRow.session_id == session_id,
            )
        )
        now = datetime.now(UTC)
        if row is None:
            row = TestChatSessionRow(
                tenant_id=self._tenant_id,
                session_id=session_id,
                label=resolved_label,
                last_quote_name=last_quote_name,
                created_at=now,
                updated_at=now,
            )
            self._session.add(row)
        else:
            row.label = resolved_label
            row.updated_at = now
            if last_quote_name is not None:
                row.last_quote_name = last_quote_name
        self._session.flush()

    def find(self, session_id: str) -> TestChatSessionRow | None:
        return self._session.scalar(
            select(TestChatSessionRow).where(
                TestChatSessionRow.tenant_id == self._tenant_id,
                TestChatSessionRow.session_id == session_id,
            )
        )

    def clear_quote(self, session_id: str) -> None:
        row = self.find(session_id)
        if row is None:
            return
        row.last_quote_name = None
        row.updated_at = datetime.now(UTC)
        self._session.flush()

    def list_recent(self, *, limit: int = 20) -> list[TestChatSessionRow]:
        rows = list(
            self._session.scalars(
                select(TestChatSessionRow)
                .where(TestChatSessionRow.tenant_id == self._tenant_id)
                .order_by(desc(TestChatSessionRow.updated_at))
                .limit(limit * 2)
            ).all()
        )
        identified = [
            row
            for row in rows
            if row.session_id.startswith(("email:", "whatsapp:"))
        ]
        return identified[:limit]

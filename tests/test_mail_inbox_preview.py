from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from chatbot.adapters.mail.imap_client import ImapMailClient, InboxPreviewMessage
from chatbot.application.mail_inbox_preview_service import _skip_reason_for_message


def test_list_recent_messages_returns_newest_first() -> None:
    client = ImapMailClient({"imap_host": "x", "username": "u"})
    client._conn = MagicMock()
    client._conn.uid.return_value = ("OK", [b"10 12 11"])
    preview = InboxPreviewMessage(
        uid="12",
        from_addr="a@b.com",
        to_addr="bot@test.local",
        subject="Hi",
        body_preview="Hello",
        received_at=datetime(2026, 6, 19, 12, 0, tzinfo=UTC),
    )
    client._parse_fetched_preview = MagicMock(return_value=preview)
    mails = client.list_recent_messages(limit=2)
    assert len(mails) == 2
    assert mails[0].uid == "12"
    assert client._conn.uid.call_args_list[0].args[2] == "ALL"


def test_skip_reason_before_process_since() -> None:
    mail = InboxPreviewMessage(
        uid="1",
        from_addr="client@example.com",
        to_addr="bot@test.local",
        subject="Old",
        body_preview="Body",
        received_at=datetime(2026, 6, 19, 6, 0, tzinfo=UTC),
    )
    reason = _skip_reason_for_message(
        mail=mail,
        process_since=datetime(2026, 6, 19, 10, 42, tzinfo=UTC),
        uid_repo=MagicMock(exists_by_uid=MagicMock(return_value=False)),
        draft_repo=MagicMock(exists_by_uid=MagicMock(return_value=False)),
    )
    assert reason is not None
    assert "before process_since" in reason


def test_skip_reason_no_body() -> None:
    mail = InboxPreviewMessage(
        uid="2",
        from_addr="client@example.com",
        to_addr="bot@test.local",
        subject="Empty",
        body_preview="(no text/plain or text/html body)",
        received_at=datetime(2026, 6, 19, 12, 0, tzinfo=UTC),
    )
    reason = _skip_reason_for_message(
        mail=mail,
        process_since=None,
        uid_repo=MagicMock(exists_by_uid=MagicMock(return_value=False)),
        draft_repo=MagicMock(exists_by_uid=MagicMock(return_value=False)),
    )
    assert reason == "no text body (worker ignores this message)"

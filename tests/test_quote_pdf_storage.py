from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evenor.application.quote_pdf_storage import (
    AttachmentValidationError,
    cleanup_pending_reply_attachments,
    delete_attachment_files,
    encode_attachments_json,
    is_user_attachment_path,
    parse_attachment_paths,
    quote_pdf_path,
    safe_quote_filename,
    store_outbound_attachment,
    store_quote_pdf,
    validate_outbound_attachment_upload,
)
from evenor.config.settings import get_settings
from evenor.domain.models.pending_reply import PendingReply, PendingReplyStatus


def test_safe_quote_filename() -> None:
    assert safe_quote_filename("QTN/0001") == "QTN-0001"


def test_store_and_resolve_pdf_path(tmp_path) -> None:
    settings = get_settings()
    settings.data_root = tmp_path
    path = store_quote_pdf(settings, "bot", "QTN-0001", b"%PDF")
    assert path.is_file()
    assert quote_pdf_path(settings, "bot", "QTN-0001") == path
    assert quote_pdf_path(settings, "bot", "missing") is None


def test_delete_attachment_files(tmp_path) -> None:
    pdf = tmp_path / "quote.pdf"
    pdf.write_bytes(b"%PDF")
    attachments_json = encode_attachments_json(
        [{"filename": "quote.pdf", "path": str(pdf), "mime_type": "application/pdf"}]
    )
    delete_attachment_files(attachments_json)
    assert not pdf.is_file()


def test_cleanup_pending_reply_attachments(tmp_path) -> None:
    pdf = tmp_path / "quote.pdf"
    pdf.write_bytes(b"%PDF")
    now = datetime.now(UTC)
    reply = PendingReply(
        id=1,
        tenant_id=1,
        connector_id=1,
        session_id="email:a@example.com",
        channel="email",
        recipient_id="a@example.com",
        draft_text="Quote",
        status=PendingReplyStatus.PENDING,
        created_at=now,
        updated_at=now,
        attachments_json=json.dumps([{"filename": "q.pdf", "path": str(pdf)}]),
    )
    cleanup_pending_reply_attachments(reply)
    assert not pdf.is_file()


def test_parse_attachment_paths_ignores_invalid_json() -> None:
    assert parse_attachment_paths(None) == []
    assert parse_attachment_paths("not-json") == []
    assert parse_attachment_paths('{"x": 1}') == []


def test_store_schedules_ttl_deletion(tmp_path, monkeypatch) -> None:
    settings = get_settings()
    settings.data_root = tmp_path
    scheduled: list[int] = []

    class FakeTimer:
        def __init__(self, delay: float, fn) -> None:
            scheduled.append(int(delay))
            self._fn = fn

        def start(self) -> None:
            self._fn()

    monkeypatch.setattr("evenor.application.quote_pdf_storage.threading.Timer", FakeTimer)
    path = store_quote_pdf(settings, "bot", "QTN-0001", b"%PDF", ttl_seconds=120)
    assert scheduled == [120]
    assert not path.is_file()


def test_store_outbound_attachment_and_path_guard(tmp_path) -> None:
    settings = get_settings()
    settings.data_root = tmp_path
    entry = store_outbound_attachment(
        settings,
        "bot",
        42,
        "note.pdf",
        b"%PDF-1.4",
        mime_type="application/pdf",
    )
    path = Path(entry["path"])
    assert path.is_file()
    assert is_user_attachment_path(settings, "bot", 42, path)
    assert not is_user_attachment_path(settings, "bot", 99, path)


def test_validate_outbound_attachment_upload_rejects_large_file(tmp_path) -> None:
    settings = get_settings()
    settings.data_root = tmp_path
    settings.attachment_max_bytes = 10
    with pytest.raises(AttachmentValidationError, match="maximum size"):
        validate_outbound_attachment_upload(
            settings,
            filename="big.pdf",
            data=b"x" * 20,
            content_type="application/pdf",
            existing_json=None,
        )

from __future__ import annotations

from sqlalchemy import select

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.conversation_repository import SqlAlchemyConversationRepository
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.orm import PendingReplyEditRow
from chatbot.adapters.mail.body_format import email_draft_html_from_markdown, sanitize_email_html
from chatbot.adapters.persistence.pending_reply_repository import SqlAlchemyPendingReplyRepository
from chatbot.application.draft_edit_service import draft_edit_text_diff, save_pending_reply_draft
from chatbot.application.validation_audit_service import ValidationAuditService
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.message import ChatMessage, MessageRole


def _spurious_diff_pairs(diff: str) -> list[tuple[str, str]]:
    lines = diff.splitlines()
    pairs: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        if not line.startswith("-") or line.startswith("---"):
            continue
        if i + 1 >= len(lines):
            continue
        nxt = lines[i + 1]
        if not nxt.startswith("+") or nxt.startswith("+++"):
            continue
        if line[1:].strip() == nxt[1:].strip():
            pairs.append((line, nxt))
    return pairs


def test_save_pending_reply_draft_logs_diff_and_syncs_message(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="**Hello** client",
            draft_html="<p><strong>Hello</strong> client</p>",
        )
        conv = SqlAlchemyConversationRepository(session, tenant.id)
        conv.append_message(
            "email:client@example.com",
            ChatMessage(role=MessageRole.ASSISTANT, content="**Hello** client"),
        )
        session.commit()
        pending_id = pending.id

    with factory() as session:
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(pending_id)
        assert reply is not None
        updated = save_pending_reply_draft(
            session,
            tenant_id=tenant.id,
            reply=reply,
            draft_html="<p>Hello <em>edited</em> client</p>",
            edited_by="admin@example.com",
        )
        session.commit()

    assert updated.draft_html is not None
    assert "<em>edited</em>" in updated.draft_html
    assert "edited" in updated.draft_text

    with factory() as session:
        messages = SqlAlchemyConversationRepository(session, tenant.id).list_messages(
            "email:client@example.com"
        )
        assert len(messages) == 1
        assert "edited" in messages[0].content
        edits = session.scalars(select(PendingReplyEditRow)).all()
        assert len(edits) == 1
        assert edits[0].edited_by == "admin@example.com"
        assert edits[0].diff
        assert "<p>" not in edits[0].diff


def test_draft_edit_text_diff_shows_targeted_markdown_change(test_settings, test_tenant) -> None:
    before_md = """Dear Mr. Moonien,

Please find below our quotation:

**Quotation for Chartreuse Group**

- **22 x SF 300 Clockers:** MUR 8,500.00
- **1 x BioTime Upgrade:** MUR 8,500.00

**Total: MUR 290,625.98**

Sincerely,
VDtec"""
    before_html = email_draft_html_from_markdown(before_md)
    after_html = (
        "<p>Dear&nbsp;Mr.&nbsp;Moonien,</p>"
        "<p>Please&nbsp;find&nbsp;below&nbsp;our&nbsp;quotation:</p>"
        "<p><strong>THIS&nbsp;IS&nbsp;SAMUEL&#39;S&nbsp;EDIT&nbsp;TEST</strong></p>"
        "<p><strong>Total:&nbsp;MUR&nbsp;290,625.98</strong></p>"
        "<p>Sincerely,<br>VDtec</p>"
    )

    diff = draft_edit_text_diff(before_html, after_html)
    minus_lines = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    plus_lines = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]

    assert any("SAMUEL" in line for line in plus_lines)
    assert any("22 x SF 300" in line for line in minus_lines)
    assert len(minus_lines) < 8
    assert not _spurious_diff_pairs(diff)


def test_draft_edit_text_diff_second_edit_only_shows_changed_blocks() -> None:
    before_md = """Dear Mr. Moonien,

Thank you for your interest and trust in VDtec.

Following our discussion and your request, we are pleased to provide you with a proposal for the replacement of your SKBL clockers with VDtec systems, including BioTime licensing and API integration with Sicorax.

Here is our estimated quotation based on your requirements:

**Quotation Details:**

* **SF 300 ZKTECO Clockers:** 22 units x 8,500 Rs = 187,000 Rs
* **BioTime License Upgrade:** 1 unit x 8,500 Rs = 8,500 Rs

**Sub-total (excluding Maintenance Contract): 264,500 Rs**

*This quotation is an estimate based on the information provided.*

**Regarding your proposed deployment approach:**

We find your approach very relevant and efficient for minimizing operational disruptions.

* **Creation of staff profiles and access rights at Head Office:** This method is entirely feasible.

We remain at your disposal for any questions or to arrange a technical visit.

Sincerely,

**VDtec Distributors Ltd**  
Office 101, Ebene Junction, Rue de la Démocratie, Ebene, Mauritius  
Tel: (+230) 464 1716 | Mobile: (+230) 5 421 1715  
Email: sales@vdtec.net | Web: www.vdtec.net"""

    before_html = sanitize_email_html(email_draft_html_from_markdown(before_md))

    after_lines: list[str] = []
    skip = False
    for line in before_md.splitlines():
        if line.strip() == "**Quotation Details:**":
            after_lines.append("**SAMUEL TEST**")
            skip = True
            continue
        if skip and line.strip().startswith("**Regarding your proposed"):
            skip = False
        if skip:
            continue
        after_lines.append(line)
    after_html = sanitize_email_html(email_draft_html_from_markdown("\n".join(after_lines)))

    diff = draft_edit_text_diff(before_html, after_html)
    minus_lines = [line for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    plus_lines = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]

    assert any("SAMUEL" in line for line in plus_lines)
    assert any("Quotation Details" in line or "SF 300" in line for line in minus_lines)
    assert "Dear Mr. Moonien," not in "".join(minus_lines)
    assert "Dear Mr. Moonien," not in "".join(plus_lines)
    assert not _spurious_diff_pairs(diff)


def test_validation_timeline_recomputes_markdown_diff(test_settings, test_tenant) -> None:
    tenant, _token = test_tenant
    engine = create_db_engine(test_settings, for_tests=True)
    factory = session_factory(engine)

    before_html = email_draft_html_from_markdown("Hello\n\n- item one\n- item two")
    after_html = "<p>Hello</p><p><strong>THIS&nbsp;IS&nbsp;SAMUEL&#39;S&nbsp;EDIT&nbsp;TEST</strong></p>"

    with factory() as session:
        connector = SqlAlchemyConnectorRepository(session).create(
            tenant_id=tenant.id,
            direction=ConnectorDirection.OUT,
            type=ConnectorType.EMAIL,
            mode=ConnectorMode.VALIDATION,
            config={"from_addr": "bot@test.local"},
        )
        pending = SqlAlchemyPendingReplyRepository(session).create(
            tenant_id=tenant.id,
            connector_id=connector.id,
            session_id="email:client@example.com",
            channel="email",
            recipient_id="client@example.com",
            draft_text="Hello",
            draft_html=before_html,
        )
        session.commit()
        pending_id = pending.id

    with factory() as session:
        reply = SqlAlchemyPendingReplyRepository(session).find_by_id(pending_id)
        assert reply is not None
        save_pending_reply_draft(
            session,
            tenant_id=tenant.id,
            reply=reply,
            draft_html=after_html,
            edited_by="admin@example.com",
        )
        session.commit()

    with factory() as session:
        timeline = ValidationAuditService(session).list_timeline_for_reply(tenant.id, pending_id)
        edit_entries = [entry for entry in timeline if entry.action == "edit"]
        assert len(edit_entries) == 1
        assert edit_entries[0].diff is not None
        assert "SAMUEL" in edit_entries[0].diff
        assert "<p>" not in edit_entries[0].diff

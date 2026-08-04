from __future__ import annotations

import pytest

from evenor.application.email_outbound import coalesce_stored_email_subject, resolve_email_subject


@pytest.mark.parametrize(
    ("draft_subject", "connector_config", "inbound_subject", "expected"),
    [
        ("Custom subject", {}, "Original", "Custom subject"),
        (None, {"default_subject": "Support reply"}, "Original", "Support reply"),
        (None, {}, "Need pricing", "Re: Need pricing"),
        (None, {}, "Re: Need pricing", "Re: Need pricing"),
        (None, {}, "RE: Need pricing", "RE: Need pricing"),
        (None, {}, "", "Reply"),
        ("  ", {"default_subject": "  "}, "Hello", "Re: Hello"),
    ],
)
def test_resolve_email_subject(
    draft_subject: str | None,
    connector_config: dict,
    inbound_subject: str,
    expected: str,
) -> None:
    assert (
        resolve_email_subject(
            draft_subject=draft_subject,
            connector_config=connector_config,
            inbound_subject=inbound_subject or None,
        )
        == expected
    )


def test_coalesce_stored_email_subject_ignores_generic_reply_when_inbound_present() -> None:
    assert (
        coalesce_stored_email_subject(
            stored_draft_subject="Reply",
            inbound_subject="Re: EBOP LTD",
        )
        is None
    )
    assert (
        resolve_email_subject(
            draft_subject=coalesce_stored_email_subject(
                stored_draft_subject="Reply",
                inbound_subject="Re: EBOP LTD",
            ),
            inbound_subject="Re: EBOP LTD",
        )
        == "Re: EBOP LTD"
    )


def test_coalesce_keeps_operator_edited_subject() -> None:
    assert (
        coalesce_stored_email_subject(
            stored_draft_subject="Custom follow-up",
            inbound_subject="Re: EBOP LTD",
        )
        == "Custom follow-up"
    )

from __future__ import annotations

import pytest

from chatbot.application.email_outbound import resolve_email_subject


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

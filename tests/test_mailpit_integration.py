"""Optional live Mailpit integration — skipped unless MAILPIT_SMTP is set."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("MAILPIT_SMTP"), reason="Set MAILPIT_SMTP=host:port to run")
def test_mailpit_accepts_smtp() -> None:
    host_port = os.environ["MAILPIT_SMTP"]
    host, port_str = host_port.rsplit(":", 1)
    msg = EmailMessage()
    msg["From"] = "bot@test.local"
    msg["To"] = "client@example.com"
    msg["Subject"] = "Outbound test"
    msg.set_content("Approved reply body")
    with smtplib.SMTP(host, int(port_str), timeout=10) as smtp:
        smtp.send_message(msg)

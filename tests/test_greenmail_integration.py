"""Optional live GreenMail integration — skipped unless GREENMAIL_SMTP is set."""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("GREENMAIL_SMTP"), reason="Set GREENMAIL_SMTP=host:port to run")
def test_greenmail_accepts_smtp_inject() -> None:
    host_port = os.environ["GREENMAIL_SMTP"]
    host, port_str = host_port.rsplit(":", 1)
    msg = EmailMessage()
    msg["From"] = "client@example.com"
    msg["To"] = "bot@test.local"
    msg["Subject"] = "Integration test"
    msg.set_content("Hello from pytest")
    with smtplib.SMTP(host, int(port_str), timeout=10) as smtp:
        smtp.send_message(msg)

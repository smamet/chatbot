from __future__ import annotations

import hashlib
import hmac

from chatbot.adapters.channels import whatsapp_meta


def test_verify_signature_accepts_valid() -> None:
    secret = "mysecret"
    body = b'{"hello":"world"}'
    sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert whatsapp_meta.verify_signature(body, sig, secret) is True


def test_verify_signature_rejects_bad() -> None:
    secret = "mysecret"
    body = b'{"hello":"world"}'
    assert whatsapp_meta.verify_signature(body, "sha256=deadbeef", secret) is False

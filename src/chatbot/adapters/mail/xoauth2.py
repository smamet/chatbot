from __future__ import annotations


def build_xoauth2_string(username: str, access_token: str) -> str:
    """Raw SASL XOAUTH2 payload (imaplib/smtplib base64-encode it)."""
    return f"user={username}\x01auth=Bearer {access_token}\x01\x01"

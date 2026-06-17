from __future__ import annotations

import base64


def build_xoauth2_string(username: str, access_token: str) -> bytes:
    raw = f"user={username}\x01auth=Bearer {access_token}\x01\x01"
    return base64.b64encode(raw.encode("utf-8"))

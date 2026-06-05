from __future__ import annotations

import json
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from chatbot.config.settings import get_settings


def _fernet() -> Fernet:
    key = get_settings().app_secret_key.strip()
    if not key:
        raise RuntimeError("APP_SECRET_KEY is not configured")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt_json(data: dict[str, Any]) -> str:
    raw = json.dumps(data, ensure_ascii=True).encode("utf-8")
    return _fernet().encrypt(raw).decode("ascii")


def decrypt_json(blob: str | None) -> dict[str, Any]:
    if not blob or not blob.strip():
        return {}
    try:
        raw = _fernet().decrypt(blob.encode("ascii"))
    except InvalidToken:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def encrypt_text(plain: str) -> str:
    if not plain:
        return ""
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt_text(blob: str | None) -> str:
    if not blob or not blob.strip():
        return ""
    try:
        return _fernet().decrypt(blob.encode("ascii")).decode("utf-8")
    except InvalidToken:
        return ""

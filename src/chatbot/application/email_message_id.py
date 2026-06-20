from __future__ import annotations

import hashlib
import re
from uuid import uuid4

_MESSAGE_ID_RE = re.compile(r"<[^>]+>")


def normalize_message_id(raw: str | None) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("<") and text.endswith(">"):
        return text
    return f"<{text.strip('<>')}>"


def parse_references_header(header: str | None) -> tuple[str, ...]:
    if not header:
        return ()
    return tuple(_MESSAGE_ID_RE.findall(header))


def make_thread_key(
    *,
    root_message_id: str | None,
    normalized_subject: str,
    received_date_iso: str,
) -> str:
    seed = (root_message_id or "").strip() or f"{normalized_subject}:{received_date_iso}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def generate_message_id(from_addr: str) -> str:
    domain = from_addr.split("@")[-1].strip().lower() if "@" in from_addr else "localhost"
    return f"<{uuid4().hex}@{domain}>"


def build_references_header(*parts: str | None) -> str | None:
    seen: list[str] = []
    for part in parts:
        mid = normalize_message_id(part)
        if mid and mid not in seen:
            seen.append(mid)
    if not seen:
        return None
    return " ".join(seen)

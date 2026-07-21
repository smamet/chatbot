from __future__ import annotations

_THREAD_SEP = "~"


def build_email_thread_session_id(from_addr: str, thread_key: str) -> str:
    email = from_addr.strip().lower()
    key = thread_key.strip()
    if not email or not key:
        raise ValueError("from_addr and thread_key are required")
    return f"email:{email}{_THREAD_SEP}{key}"


def parse_email_thread_session_id(session_id: str) -> tuple[str | None, str | None]:
    if not session_id.startswith("email:"):
        return None, None
    raw = session_id.removeprefix("email:").strip()
    if not raw:
        return None, None
    email_part, sep, rest = raw.partition("|")
    if sep:
        email_part, _, thread_part = email_part.partition(_THREAD_SEP)
        email = email_part.strip().lower() or None
        thread_key = thread_part.strip() or None
        return email, thread_key
    if _THREAD_SEP in email_part:
        email, _, thread_key = email_part.partition(_THREAD_SEP)
        return email.strip().lower() or None, thread_key.strip() or None
    return email_part.strip().lower() or None, None


def strip_thread_from_email_session_part(raw_id: str) -> str:
    """Return email(+optional phone) portion without ~thread_key."""
    email_phone, sep, phone = raw_id.partition("|")
    email_only, thread_sep, thread_key = email_phone.partition(_THREAD_SEP)
    if thread_sep and thread_key and not phone:
        return email_only
    if thread_sep and thread_key and phone:
        return f"{email_only}|{phone}"
    return raw_id

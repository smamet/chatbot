from __future__ import annotations

import re

_ERPNEXT_DATETIME_MICROSECONDS = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+"
)


def format_erpnext_error_message(message: str | None) -> str:
    if not message:
        return ""
    return _ERPNEXT_DATETIME_MICROSECONDS.sub(r"\1", message)

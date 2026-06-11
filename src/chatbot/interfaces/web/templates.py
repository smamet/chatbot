from __future__ import annotations

import json
from pathlib import Path

from fastapi.templating import Jinja2Templates

from chatbot.application.context_debug import format_context_debug_label
from chatbot.application.customer_access_gate import (
    resolve_manual_identity,
    session_display_label,
    session_resume_params,
    session_test_chat_query,
)
from chatbot.mail.process_since import format_process_since_display


def pretty_json(value: str | None) -> str:
    if not value:
        return ""
    try:
        parsed = json.loads(value)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (TypeError, json.JSONDecodeError):
        return str(value)


TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["session_label"] = session_display_label
templates.env.filters["session_resume_params"] = session_resume_params
templates.env.filters["session_test_chat_query"] = session_test_chat_query
templates.env.filters["context_debug_label"] = format_context_debug_label
templates.env.filters["process_since_display"] = format_process_since_display
templates.env.filters["pretty_json"] = pretty_json

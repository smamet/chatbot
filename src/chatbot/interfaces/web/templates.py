from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from chatbot.application.customer_access_gate import session_display_label
from chatbot.mail.process_since import format_process_since_display

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["session_label"] = session_display_label
templates.env.filters["process_since_display"] = format_process_since_display

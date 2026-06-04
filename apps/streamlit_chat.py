from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import httpx
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_repo_dotenv() -> None:
    """Streamlit does not load `.env` by default; the API does via pydantic-settings."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


_load_repo_dotenv()

DEFAULT_API = "http://127.0.0.1:8000"


def _chat_api_secret() -> str:
    v = (os.environ.get("CHAT_API_SECRET") or "").strip()
    if v:
        return v
    try:
        s = st.secrets["CHAT_API_SECRET"]
        return str(s).strip() if s is not None else ""
    except (FileNotFoundError, KeyError, TypeError, RuntimeError):
        return ""


def _chat_request_headers() -> dict[str, str]:
    secret = _chat_api_secret()
    if not secret:
        return {}
    return {"Authorization": f"Bearer {secret}"}


def _human_error_message(body: object) -> str:
    """Pick a single user-facing string from common API error JSON shapes."""
    if not isinstance(body, dict):
        return str(body)
    detail = body.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, dict):
        msg = detail.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
        err = detail.get("error")
        if isinstance(err, dict):
            nested = err.get("message")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    err = body.get("error")
    if isinstance(err, dict):
        nested = err.get("message")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    try:
        return json.dumps(body, ensure_ascii=False)[:500]
    except (TypeError, ValueError):
        return str(body)[:500]


def _human_error_message_from_response(r: httpx.Response) -> str:
    try:
        body = r.json()
    except json.JSONDecodeError:
        return (r.text or r.reason_phrase or "Unknown error").strip()[:2000]
    return _human_error_message(body)


st.set_page_config(page_title="Chatbot test", layout="centered")
st.title("Chatbot test client")

api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API)
if not _chat_api_secret():
    st.sidebar.caption(
        "If the API uses CHAT_API_SECRET: set it in project `.env`, export it, or use `.streamlit/secrets.toml`."
    )
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
st.sidebar.caption(f"session_id: {st.session_state.session_id}")
if st.sidebar.button("New session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pop("last_usage", None)
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

usage = st.session_state.get("last_usage")
if usage:
    st.sidebar.caption(f"last usage: {usage}")

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

uploaded_files = st.file_uploader(
    "Attachments",
    accept_multiple_files=True,
    key="chat_attachments",
)

prompt = st.chat_input("Message")
if prompt:
    url = f"{api_base.rstrip('/')}/v1/chat"
    form_data = {
        "session_id": st.session_state.session_id,
        "message": prompt,
    }
    file_parts: list[tuple[str, tuple[str, bytes, str]]] = []
    if uploaded_files:
        for uf in uploaded_files:
            mime = uf.type or "application/octet-stream"
            file_parts.append(("files", (uf.name, uf.getvalue(), mime)))
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                url,
                data=form_data,
                files=file_parts or None,
                headers=_chat_request_headers(),
            )
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        msg = _human_error_message_from_response(e.response) if e.response is not None else str(e)
        st.error(msg)
        st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()
    reply = data.get("reply", "")
    st.session_state.last_usage = data.get("usage") or {}
    user_display = prompt
    if uploaded_files:
        names = ", ".join(uf.name for uf in uploaded_files)
        user_display = f"{prompt}\n\n*(attachments: {names})*"
    st.session_state.messages.append(("user", user_display))
    st.session_state.messages.append(("assistant", reply))
    st.session_state.pop("chat_attachments", None)
    st.rerun()

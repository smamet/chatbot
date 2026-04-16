from __future__ import annotations

import json
import uuid

import httpx
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"


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

prompt = st.chat_input("Message")
if prompt:
    url = f"{api_base.rstrip('/')}/v1/chat"
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                url,
                json={"session_id": st.session_state.session_id, "message": prompt},
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
    st.session_state.messages.append(("user", prompt))
    st.session_state.messages.append(("assistant", reply))
    st.rerun()

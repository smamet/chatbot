from __future__ import annotations

from pydantic import BaseModel

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from chatbot.application.chat_service import ChatService
from chatbot.config.settings import Settings
from chatbot.domain.models.attachment import Attachment
from chatbot.interfaces.api.deps import get_chat_service, get_settings_dep, require_chat_api_auth

router = APIRouter()


class UsageOut(BaseModel):
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    reply: str
    usage: UsageOut


@router.post("/chat", response_model=ChatResponse)
async def post_chat(
    session_id: str = Form(..., min_length=1, max_length=256),
    message: str = Form(..., min_length=1),
    files: list[UploadFile] = File(default=[]),
    _: None = Depends(require_chat_api_auth),
    service: ChatService = Depends(get_chat_service),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponse:
    attachments: list[Attachment] | None = None
    if files:
        attachments = []
        for f in files:
            data = await f.read()
            mime = f.content_type or "application/octet-stream"
            attachments.append(
                Attachment(mime_type=mime, data=data, filename=f.filename)
            )
    try:
        result = service.handle_user_message(
            session_id, message, attachments=attachments
        )
    except Exception as e:
        if not settings.dev_mode:
            raise HTTPException(status_code=500, detail="Internal server error") from e
        raise HTTPException(
            status_code=500,
            detail={
                "kind": "internal",
                "type": type(e).__name__,
                "message": str(e),
            },
        ) from e
    u = result.usage
    return ChatResponse(
        reply=result.text,
        usage=UsageOut(
            prompt_tokens=u.prompt_tokens,
            candidates_tokens=u.candidates_tokens,
            total_tokens=u.total_tokens,
        ),
    )

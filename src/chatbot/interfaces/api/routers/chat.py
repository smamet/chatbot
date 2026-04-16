from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException

from chatbot.application.chat_service import ChatService
from chatbot.config.settings import Settings
from chatbot.interfaces.api.deps import get_chat_service, get_settings_dep

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=256)
    message: str = Field(..., min_length=1)


class UsageOut(BaseModel):
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    reply: str
    usage: UsageOut


@router.post("/chat", response_model=ChatResponse)
def post_chat(
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponse:
    try:
        result = service.handle_user_message(body.session_id, body.message)
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

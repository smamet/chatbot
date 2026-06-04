from __future__ import annotations

from pathlib import Path

from chatbot.application.order_command_extractor import extract_order_command
from chatbot.application.order_service import OrderService
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.config.settings import Settings
from chatbot.domain.contracts.conversation_repository import ConversationRepository
from chatbot.domain.contracts.llm_client import LlmClient, LlmResult
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LlmClient,
        repo: ConversationRepository,
        rag: RagPipeline | None,
        order_service: OrderService | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._repo = repo
        self._rag = rag
        self._order_service = order_service
        self._prompt_path = prompt_path or settings.prompt_path

    def _load_system_instruction(self) -> str:
        path = self._prompt_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return "You are a helpful assistant."

    @staticmethod
    def _content_with_attachment_notes(
        user_message: str, attachments: list[Attachment] | None
    ) -> str:
        if not attachments:
            return user_message
        lines = [user_message] if user_message.strip() else []
        for att in attachments:
            label = att.filename or att.mime_type
            lines.append(f"[Attached: {label}]")
        return "\n".join(lines)

    def handle_user_message(
        self,
        session_id: str,
        user_message: str,
        *,
        attachments: list[Attachment] | None = None,
    ) -> LlmResult:
        content = self._content_with_attachment_notes(user_message, attachments)
        user_msg = ChatMessage(role=MessageRole.USER, content=content)
        self._repo.append_message(session_id, user_msg)
        history = self._repo.list_messages(session_id, limit=50)
        system = self._load_system_instruction()
        if self._rag and self._settings.rag_enabled:
            ctx = self._rag.build_retrieval_context(user_message)
            if ctx:
                system = f"{system}\n\n--- Retrieved context ---\n{ctx}"
                if not self._settings.dev_mode:
                    system = (
                        f"{system}\n\n"
                        "Do not mention internal file names, paths, or parenthetical "
                        "source citations such as (Source: …) in your reply to the customer."
                    )
        result = self._llm.generate_chat(
            system_instruction=system,
            messages=history,
            attachments=attachments,
        )
        extracted = extract_order_command(result.text)
        self._repo.append_message(
            session_id,
            ChatMessage(role=MessageRole.ASSISTANT, content=extracted.clean_reply),
        )
        if self._order_service and extracted.command:
            context = self._repo.list_messages(session_id, limit=6)
            self._order_service.append_command(
                session_id=session_id,
                command=extracted.command,
                command_json=extracted.command_json,
                conversation_context=context,
            )
        return LlmResult(text=extracted.clean_reply, usage=result.usage)

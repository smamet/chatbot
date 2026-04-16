from __future__ import annotations

from pathlib import Path

from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.config.settings import Settings
from chatbot.domain.contracts.conversation_repository import ConversationRepository
from chatbot.domain.contracts.llm_client import LlmClient, LlmResult
from chatbot.domain.models.message import ChatMessage, MessageRole


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LlmClient,
        repo: ConversationRepository,
        rag: RagPipeline | None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._repo = repo
        self._rag = rag
        self._prompt_path = prompt_path or settings.prompt_path

    def _load_system_instruction(self) -> str:
        path = self._prompt_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return "You are a helpful assistant."

    def handle_user_message(self, session_id: str, user_message: str) -> LlmResult:
        user_msg = ChatMessage(role=MessageRole.USER, content=user_message)
        self._repo.append_message(session_id, user_msg)
        history = self._repo.list_messages(session_id, limit=50)
        system = self._load_system_instruction()
        if self._rag and self._settings.rag_enabled:
            ctx = self._rag.build_retrieval_context(user_message)
            if ctx:
                system = f"{system}\n\n--- Retrieved context ---\n{ctx}"
        result = self._llm.generate_chat(system_instruction=system, messages=history)
        self._repo.append_message(
            session_id,
            ChatMessage(role=MessageRole.ASSISTANT, content=result.text),
        )
        return result

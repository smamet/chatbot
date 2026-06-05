from __future__ import annotations

from chatbot.application.hook_extractor import extract_hook
from chatbot.application.rag_orchestrator import RagPipeline
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.config.settings import Settings
from chatbot.domain.contracts.conversation_repository import ConversationRepository
from chatbot.domain.contracts.hook_event_repository import HookEventRepository
from chatbot.domain.contracts.llm_client import LlmClient, LlmResult
from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage, MessageRole
from chatbot.domain.models.tenant import Tenant


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        tenant: Tenant,
        llm: LlmClient,
        repo: ConversationRepository,
        rag: RagPipeline | None,
        hook_repo: HookEventRepository | None = None,
    ) -> None:
        self._global_settings = settings
        self._tenant = tenant
        self._settings = merge_tenant_settings(settings, tenant)
        self._llm = llm
        self._repo = repo
        self._rag = rag
        self._hook_repo = hook_repo

    def _load_system_instruction(self) -> str:
        parts: list[str] = []
        prompt = (self._tenant.prompt or "").strip()
        if prompt:
            parts.append(prompt)
        if self._tenant.hooks_enabled:
            hook = (self._tenant.hook_instructions or "").strip()
            if hook:
                parts.append(hook)
        return "\n\n".join(parts) if parts else "You are a helpful assistant."

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
        extracted = extract_hook(result.text)
        self._repo.append_message(
            session_id,
            ChatMessage(role=MessageRole.ASSISTANT, content=extracted.clean_reply),
        )
        if (
            self._hook_repo
            and self._tenant.hooks_enabled
            and extracted.hook_type
            and extracted.payload_json
        ):
            self._hook_repo.create(
                session_id=session_id,
                hook_type=extracted.hook_type,
                payload_json=extracted.payload_json,
            )
        return LlmResult(text=extracted.clean_reply, usage=result.usage)

    def draft_reply(self, session_id: str, inbound_text: str) -> LlmResult:
        return self.handle_user_message(session_id, inbound_text)

from __future__ import annotations

from collections.abc import Callable

from evenor.application.hook_extractor import extract_hook
from evenor.application.hook_prompt_composer import compose_hook_instructions, hooks_enabled_for_tenant
from evenor.application.rag_orchestrator import RagPipeline
from evenor.application.tenant_settings import merge_tenant_settings
from evenor.config.settings import Settings
from evenor.domain.contracts.conversation_repository import ConversationRepository
from evenor.domain.contracts.hook_event_repository import HookEventRepository
from evenor.domain.contracts.llm_client import LlmClient, LlmResult
from evenor.domain.models.context_debug import ContextDebugInfo
from evenor.domain.models.attachment import Attachment
from evenor.domain.models.message import ChatMessage, MessageRole
from evenor.domain.models.tenant import Tenant


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
        integration_enricher: Callable[[str], str | None] | None = None,
        erp_enricher: Callable[[str], str | None] | None = None,
        active_integrations: set[str] | None = None,
    ) -> None:
        self._global_settings = settings
        self._tenant = tenant
        self._settings = merge_tenant_settings(settings, tenant)
        self._llm = llm
        self._repo = repo
        self._rag = rag
        self._hook_repo = hook_repo
        self._integration_enricher = integration_enricher or erp_enricher
        self._active_integrations = active_integrations

    def _load_system_instruction(self) -> str:
        parts: list[str] = []
        prompt = (self._tenant.prompt or "").strip()
        if prompt:
            parts.append(prompt)
        if hooks_enabled_for_tenant(self._tenant):
            hook = compose_hook_instructions(
                self._tenant,
                active_integrations=self._active_integrations,
            ).strip()
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
        customer_chars = 0
        if self._integration_enricher:
            data_block = self._integration_enricher(session_id)
            if data_block:
                customer_chars = len(data_block)
                system = f"{system}\n\n--- Customer data ---\n{data_block}"
        rag_chunks = 0
        rag_chars = 0
        if self._rag and self._settings.rag_enabled:
            retrieval = self._rag.build_retrieval_context(user_message)
            rag_chunks = retrieval.chunk_count
            rag_chars = retrieval.char_count
            if retrieval.text:
                system = f"{system}\n\n--- Retrieved context ---\n{retrieval.text}"
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
        context_debug = ContextDebugInfo(
            rag_chunks=rag_chunks,
            rag_chars=rag_chars,
            customer_chars=customer_chars,
            system_chars=len(system),
        )
        self._repo.append_message(
            session_id,
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=extracted.clean_reply,
                context_debug=context_debug,
            ),
        )
        hook_event_id: int | None = None
        if (
            self._hook_repo
            and hooks_enabled_for_tenant(self._tenant)
            and extracted.hook_type
            and extracted.payload_json
        ):
            hook_event = self._hook_repo.create(
                session_id=session_id,
                hook_type=extracted.hook_type,
                payload_json=extracted.payload_json,
            )
            hook_event_id = hook_event.id
        return LlmResult(
            text=extracted.clean_reply,
            usage=result.usage,
            hook_type=extracted.hook_type,
            hook_payload_json=extracted.payload_json,
            hook_event_id=hook_event_id,
            context_debug=context_debug,
        )

    def draft_reply(self, session_id: str, inbound_text: str) -> LlmResult:
        return self.handle_user_message(session_id, inbound_text)

    def regenerate_assistant_reply(
        self,
        session_id: str,
        *,
        history: list[ChatMessage],
        inbound_text: str,
    ) -> LlmResult:
        """Re-run the LLM for a pending reply without persisting new messages."""
        _ = session_id
        user_msg = ChatMessage(role=MessageRole.USER, content=inbound_text.strip())
        messages = [*history, user_msg]
        system = self._load_system_instruction()
        customer_chars = 0
        if self._integration_enricher:
            data_block = self._integration_enricher(session_id)
            if data_block:
                customer_chars = len(data_block)
                system = f"{system}\n\n--- Customer data ---\n{data_block}"
        rag_chunks = 0
        rag_chars = 0
        if self._rag and self._settings.rag_enabled:
            retrieval = self._rag.build_retrieval_context(inbound_text)
            rag_chunks = retrieval.chunk_count
            rag_chars = retrieval.char_count
            if retrieval.text:
                system = f"{system}\n\n--- Retrieved context ---\n{retrieval.text}"
                if not self._settings.dev_mode:
                    system = (
                        f"{system}\n\n"
                        "Do not mention internal file names, paths, or parenthetical "
                        "source citations such as (Source: …) in your reply to the customer."
                    )
        result = self._llm.generate_chat(
            system_instruction=system,
            messages=messages,
        )
        extracted = extract_hook(result.text)
        context_debug = ContextDebugInfo(
            rag_chunks=rag_chunks,
            rag_chars=rag_chars,
            customer_chars=customer_chars,
            system_chars=len(system),
        )
        return LlmResult(
            text=extracted.clean_reply,
            usage=result.usage,
            hook_type=extracted.hook_type,
            hook_payload_json=extracted.payload_json,
            hook_event_id=None,
            context_debug=context_debug,
        )

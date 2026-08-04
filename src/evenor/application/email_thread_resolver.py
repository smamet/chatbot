from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from evenor.adapters.persistence.email_thread_repository import SqlAlchemyEmailThreadRepository
from evenor.adapters.persistence.mail_draft_repository import SqlAlchemyMailDraftRepository
from evenor.adapters.persistence.outbound_email_message_repository import (
    SqlAlchemyOutboundEmailMessageRepository,
)
from evenor.application.email_message_id import make_thread_key, normalize_message_id
from evenor.application.email_subject import normalize_subject, subject_similarity
from evenor.application.email_thread_disambiguator import EmailThreadDisambiguator
from evenor.application.email_thread_resolution import (
    ThreadResolutionAudit,
    ThreadResolutionLlmMeta,
)
from evenor.config.settings import Settings
from evenor.domain.models.email_thread import EmailThread


@dataclass(frozen=True, slots=True)
class InboundEmailHeaders:
    message_id: str = ""
    in_reply_to: str = ""
    references: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedEmailThread:
    thread: EmailThread
    thread_key: str
    created: bool
    audit: ThreadResolutionAudit


class EmailThreadResolver:
    def __init__(
        self,
        session: Session,
        *,
        tenant_id: int,
        settings: Settings,
        disambiguator: EmailThreadDisambiguator | None = None,
    ) -> None:
        self._session = session
        self._tenant_id = tenant_id
        self._settings = settings
        self._threads = SqlAlchemyEmailThreadRepository(session, tenant_id=tenant_id)
        self._drafts = SqlAlchemyMailDraftRepository(session, tenant_id=tenant_id)
        self._outbound = SqlAlchemyOutboundEmailMessageRepository(session, tenant_id=tenant_id)
        self._disambiguator = disambiguator

    def resolve(
        self,
        *,
        from_addr: str,
        subject: str,
        body_new: str,
        received_at: datetime | None,
        headers: InboundEmailHeaders,
    ) -> ResolvedEmailThread:
        sender = from_addr.strip().lower()
        normalized = normalize_subject(subject)
        when = received_at or datetime.now(UTC)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        steps: list[str] = []

        in_reply_to = normalize_message_id(headers.in_reply_to)
        refs = tuple(normalize_message_id(r) for r in headers.references if r)
        inbound_mid = normalize_message_id(headers.message_id)

        thread_by_header = self._resolve_by_headers(in_reply_to, refs)
        if thread_by_header is not None:
            steps.append("rfc_headers")
            self._threads.touch_activity(thread_by_header.id, when)
            return self._resolved(
                thread=thread_by_header,
                thread_key=thread_by_header.thread_key,
                created=False,
                method="rfc_headers",
                steps=steps,
            )
        steps.append("rfc_headers:miss")

        root_message_id = refs[0] if refs else inbound_mid or None
        stale_cutoff = when - timedelta(days=max(1, self._settings.email_thread_stale_days))
        open_threads = self._threads.list_open_by_sender(sender, since=stale_cutoff)

        exact = [t for t in open_threads if t.normalized_subject == normalized and normalized]
        if len(exact) == 1:
            steps.append("subject_exact")
            self._threads.touch_activity(exact[0].id, when)
            return self._resolved(
                thread=exact[0],
                thread_key=exact[0].thread_key,
                created=False,
                method="subject_exact",
                steps=steps,
            )
        steps.append("subject_exact:miss")

        similar = [
            t
            for t in open_threads
            if subject_similarity(t.normalized_subject, normalized)
            >= self._settings.email_thread_subject_similarity
        ]

        ambiguous = len(similar) > 1 or (
            in_reply_to
            and similar
            and any(t.normalized_subject != normalized for t in similar)
        )

        llm_meta: ThreadResolutionLlmMeta | None = None
        used_llm = False
        if ambiguous:
            steps.append("subject_ambiguous")
            if self._disambiguator is not None:
                decision = self._disambiguator.disambiguate(
                    inbound_subject=normalized,
                    body_preview=body_new,
                    candidates=[
                        {
                            "thread_key": t.thread_key,
                            "subject": t.normalized_subject,
                            "last_activity": t.last_activity_at.isoformat(),
                        }
                        for t in (similar or open_threads)[:3]
                    ],
                )
                if decision.llm_called:
                    used_llm = True
                    llm_meta = ThreadResolutionLlmMeta(
                        confidence=decision.confidence,
                        prompt_tokens=decision.prompt_tokens,
                        output_tokens=decision.output_tokens,
                    )
                if decision.same_thread and decision.thread_key:
                    found = self._threads.find_by_key(sender, decision.thread_key)
                    if found is not None:
                        steps.append("llm")
                        self._threads.touch_activity(found.id, when)
                        return self._resolved(
                            thread=found,
                            thread_key=found.thread_key,
                            created=False,
                            method="llm",
                            steps=steps,
                            used_llm=used_llm,
                            llm=llm_meta,
                        )
                if used_llm:
                    low = decision.confidence < self._settings.email_thread_llm_min_confidence
                    steps.append("llm:low_confidence" if low else "llm:miss")

        if len(similar) == 1:
            steps.append("subject_similarity")
            self._threads.touch_activity(similar[0].id, when)
            return self._resolved(
                thread=similar[0],
                thread_key=similar[0].thread_key,
                created=False,
                method="subject_similarity",
                steps=steps,
                used_llm=used_llm,
                llm=llm_meta,
            )

        thread_key = make_thread_key(
            root_message_id=root_message_id,
            normalized_subject=normalized,
            received_date_iso=when.date().isoformat(),
        )
        existing = self._threads.find_by_key(sender, thread_key)
        if existing is not None:
            steps.append("thread_key_reuse")
            self._threads.touch_activity(existing.id, when)
            return self._resolved(
                thread=existing,
                thread_key=existing.thread_key,
                created=False,
                method="thread_key_reuse",
                steps=steps,
                used_llm=used_llm,
                llm=llm_meta,
            )

        created = self._threads.create(
            from_addr=sender,
            thread_key=thread_key,
            root_message_id=root_message_id,
            normalized_subject=normalized,
            last_activity_at=when,
        )
        steps.append("new_thread")
        return self._resolved(
            thread=created,
            thread_key=thread_key,
            created=True,
            method="new_thread",
            steps=steps,
            used_llm=used_llm,
            llm=llm_meta,
        )

    @staticmethod
    def _resolved(
        *,
        thread: EmailThread,
        thread_key: str,
        created: bool,
        method: str,
        steps: list[str],
        used_llm: bool = False,
        llm: ThreadResolutionLlmMeta | None = None,
    ) -> ResolvedEmailThread:
        audit = ThreadResolutionAudit(
            method=method,
            used_llm=used_llm,
            steps=tuple(steps),
            llm=llm if used_llm else None,
        )
        return ResolvedEmailThread(
            thread=thread,
            thread_key=thread_key,
            created=created,
            audit=audit,
        )

    def _resolve_by_headers(
        self,
        in_reply_to: str,
        references: tuple[str, ...],
    ) -> EmailThread | None:
        if in_reply_to:
            thread = self._thread_for_message_id(in_reply_to)
            if thread is not None:
                return thread
        for ref in reversed(references):
            thread = self._thread_for_message_id(ref)
            if thread is not None:
                return thread
        return None

    def _thread_for_message_id(self, message_id: str) -> EmailThread | None:
        mid = normalize_message_id(message_id)
        if not mid:
            return None
        outbound = self._outbound.find_by_message_id(mid)
        if outbound is not None:
            return self._threads.find_by_id(outbound.thread_id)
        draft = self._drafts.find_by_message_id(mid)
        if draft is not None and draft.thread_id is not None:
            return self._threads.find_by_id(draft.thread_id)
        return None

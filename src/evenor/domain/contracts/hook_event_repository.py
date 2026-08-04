from __future__ import annotations

from typing import Protocol

from evenor.domain.models.hook import HookEvent, HookStatus


class HookEventRepository(Protocol):
    def create(
        self,
        *,
        session_id: str,
        hook_type: str,
        payload_json: str,
    ) -> HookEvent: ...

    def list_by_tenant(
        self, *, limit: int = 100, status: HookStatus | None = None
    ) -> list[HookEvent]: ...

    def claim_pending(self, *, limit: int = 10) -> list[HookEvent]: ...

    def update_status(
        self,
        hook_id: int,
        *,
        status: HookStatus,
        error: str | None = None,
        increment_attempts: bool = False,
    ) -> None: ...

    def reset_to_pending(self, hook_id: int) -> None: ...

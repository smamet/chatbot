from __future__ import annotations

from chatbot.adapters.persistence.orm import UserRow
from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository, verify_password
from chatbot.domain.models.tenant import Tenant
from chatbot.domain.models.user import User, UserRole
from sqlalchemy import select


class UserService:
    def __init__(self, repository: SqlAlchemyUserRepository) -> None:
        self._repo = repository

    def authenticate(self, email: str, password: str) -> User | None:
        db_row = self._repo._session.scalar(
            select(UserRow).where(UserRow.email == email.lower().strip())
        )
        if db_row is None or not db_row.active:
            return None
        if not verify_password(password, db_row.password_hash):
            return None
        return self._repo.find_by_email(email)

    def can_access_tenant(self, user: User, tenant_id: int) -> bool:
        if user.role == UserRole.ADMIN:
            return True
        return tenant_id in self._repo.tenant_ids_for_user(user.id)

    def filter_tenants(self, user: User, tenants: list[Tenant]) -> list[Tenant]:
        if user.role == UserRole.ADMIN:
            return tenants
        allowed = set(self._repo.tenant_ids_for_user(user.id))
        return [t for t in tenants if t.id in allowed]

    def can_edit(self, user: User) -> bool:
        return user.role in (UserRole.ADMIN, UserRole.CLIENT_ADMIN)

    def get_by_id(self, user_id: int) -> User | None:
        return self._repo.find_by_id(user_id)

    def find_by_email(self, email: str) -> User | None:
        return self._repo.find_by_email(email)

    def list_users(self) -> list[User]:
        return self._repo.list_all()

    def create_user(self, *, email: str, password: str, role: UserRole) -> User:
        return self._repo.create(email=email, password=password, role=role)

    def set_password(self, email: str, password: str) -> User | None:
        return self._repo.set_password(email, password)

    def tenant_ids_for_user(self, user_id: int) -> list[int]:
        return self._repo.tenant_ids_for_user(user_id)

    def set_role(self, user_id: int, role: UserRole) -> User | None:
        return self._repo.set_role(user_id, role)

    def set_active(self, user_id: int, active: bool) -> User | None:
        return self._repo.set_active(user_id, active)

    def grant_access(self, user_id: int, tenant_id: int) -> None:
        self._repo.grant_bot_access(user_id, tenant_id)

    def revoke_access(self, user_id: int, tenant_id: int) -> None:
        self._repo.revoke_bot_access(user_id, tenant_id)

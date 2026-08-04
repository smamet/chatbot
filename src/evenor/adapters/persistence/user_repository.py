from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from evenor.adapters.persistence.orm import UserBotAccessRow, UserRow
from evenor.domain.models.user import User, UserRole


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return secrets.compare_digest(check, digest)


def _row_to_user(row: UserRow) -> User:
    return User(
        id=row.id,
        email=row.email,
        role=UserRole(row.role),
        active=bool(row.active),
        created_at=row.created_at.replace(tzinfo=UTC) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=UTC) if row.updated_at.tzinfo is None else row.updated_at,
    )


class SqlAlchemyUserRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def find_by_email(self, email: str) -> User | None:
        row = self._session.scalar(select(UserRow).where(UserRow.email == email.lower().strip()))
        return _row_to_user(row) if row else None

    def find_by_id(self, user_id: int) -> User | None:
        row = self._session.get(UserRow, user_id)
        return _row_to_user(row) if row else None

    def list_all(self) -> list[User]:
        return [_row_to_user(r) for r in self._session.scalars(select(UserRow).order_by(UserRow.email))]

    def set_password(self, email: str, password: str) -> User | None:
        row = self._session.scalar(select(UserRow).where(UserRow.email == email.lower().strip()))
        if row is None:
            return None
        row.password_hash = hash_password(password)
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_user(row)

    def create(self, *, email: str, password: str, role: UserRole) -> User:
        now = datetime.now(UTC)
        row = UserRow(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            role=role.value,
            active=True,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_user(row)

    def tenant_ids_for_user(self, user_id: int) -> list[int]:
        return list(
            self._session.scalars(
                select(UserBotAccessRow.tenant_id).where(UserBotAccessRow.user_id == user_id)
            )
        )

    def grant_bot_access(self, user_id: int, tenant_id: int) -> None:
        existing = self._session.scalar(
            select(UserBotAccessRow).where(
                UserBotAccessRow.user_id == user_id,
                UserBotAccessRow.tenant_id == tenant_id,
            )
        )
        if existing:
            return
        self._session.add(UserBotAccessRow(user_id=user_id, tenant_id=tenant_id))
        self._session.flush()

    def revoke_bot_access(self, user_id: int, tenant_id: int) -> None:
        row = self._session.scalar(
            select(UserBotAccessRow).where(
                UserBotAccessRow.user_id == user_id,
                UserBotAccessRow.tenant_id == tenant_id,
            )
        )
        if row:
            self._session.delete(row)
            self._session.flush()

    def set_role(self, user_id: int, role: UserRole) -> User | None:
        row = self._session.get(UserRow, user_id)
        if row is None:
            return None
        row.role = role.value
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_user(row)

    def set_active(self, user_id: int, active: bool) -> User | None:
        row = self._session.get(UserRow, user_id)
        if row is None:
            return None
        row.active = active
        row.updated_at = datetime.now(UTC)
        self._session.flush()
        self._session.refresh(row)
        return _row_to_user(row)

    def set_remember_token_hash(self, user_id: int, token_hash: str) -> None:
        row = self._session.get(UserRow, user_id)
        if row is None:
            return
        row.remember_token_hash = token_hash
        row.updated_at = datetime.now(UTC)
        self._session.flush()

    def clear_remember_token_hash(self, user_id: int) -> None:
        row = self._session.get(UserRow, user_id)
        if row is None:
            return
        row.remember_token_hash = None
        row.updated_at = datetime.now(UTC)
        self._session.flush()

    def get_remember_token_hash(self, user_id: int) -> str | None:
        row = self._session.get(UserRow, user_id)
        if row is None:
            return None
        return row.remember_token_hash

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
from chatbot.application.user_service import UserService
from chatbot.domain.models.user import User, UserRole
from chatbot.interfaces.api.deps import get_session


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(SqlAlchemyUserRepository(session))


def get_optional_user(
    request: Request,
    user_service: UserService = Depends(get_user_service),
) -> User | None:
    raw = request.session.get("user_id")
    if raw is None:
        return None
    try:
        user_id = int(raw)
    except (TypeError, ValueError):
        return None
    return user_service.get_by_id(user_id)


def require_user(user: User | None = Depends(get_optional_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def require_editor(
    user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    if not user_service.can_edit(user):
        raise HTTPException(status_code=403, detail="Read-only user")
    return user

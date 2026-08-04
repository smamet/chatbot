from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from evenor.adapters.persistence.user_repository import SqlAlchemyUserRepository
from evenor.application.remember_me_service import REMEMBER_COOKIE_NAME, RememberMeService
from evenor.application.user_service import UserService
from evenor.config.settings import Settings, get_settings
from evenor.domain.models.user import User, UserRole
from evenor.interfaces.api.deps import get_session


def get_user_service(session: Session = Depends(get_session)) -> UserService:
    return UserService(SqlAlchemyUserRepository(session))


def get_remember_me_service(session: Session = Depends(get_session)) -> RememberMeService:
    return RememberMeService(SqlAlchemyUserRepository(session), settings=get_settings())


def get_optional_user(
    request: Request,
    user_service: UserService = Depends(get_user_service),
    remember_service: RememberMeService = Depends(get_remember_me_service),
) -> User | None:
    raw = request.session.get("user_id")
    if raw is not None:
        try:
            user_id = int(raw)
        except (TypeError, ValueError):
            user_id = None
        else:
            user = user_service.get_by_id(user_id)
            if user is not None:
                return user

    signed = request.cookies.get(REMEMBER_COOKIE_NAME)
    if not signed:
        return None
    user = remember_service.authenticate_cookie(signed)
    if user is not None:
        request.session["user_id"] = user.id
    return user


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


def require_validator(
    user: User = Depends(require_user),
    user_service: UserService = Depends(get_user_service),
) -> User:
    if not user_service.can_validate(user):
        raise HTTPException(status_code=403, detail="Validation access required")
    return user


def reject_validation_only(user: User, user_service: UserService) -> None:
    if user_service.is_validation_only(user):
        raise HTTPException(status_code=403, detail="Forbidden")

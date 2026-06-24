from __future__ import annotations

from fastapi import Depends, Request
from fastapi.responses import RedirectResponse

from chatbot.application.tenant_service import TenantService
from chatbot.application.user_service import UserService
from chatbot.domain.models.user import User
from chatbot.interfaces.api.deps import get_tenant_service
from chatbot.interfaces.web.deps import get_optional_user, get_user_service


def authenticated_home_redirect(
    user: User,
    *,
    user_service: UserService,
    tenant_service: TenantService,
) -> RedirectResponse:
    home = user_service.dashboard_home_url(user, tenant_service.list_tenants())
    return RedirectResponse(url=home, status_code=302)


def resolve_authenticated_home(
    request: Request,
    user: User | None = Depends(get_optional_user),
    user_service: UserService = Depends(get_user_service),
    tenant_service: TenantService = Depends(get_tenant_service),
) -> RedirectResponse | None:
    del request
    if user is None:
        return None
    return authenticated_home_redirect(
        user,
        user_service=user_service,
        tenant_service=tenant_service,
    )

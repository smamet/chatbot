from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "admin"
    CLIENT_ADMIN = "client_admin"
    CLIENT_OPERATOR = "client_operator"


@dataclass(frozen=True)
class User:
    id: int
    email: str
    role: UserRole
    active: bool
    created_at: datetime
    updated_at: datetime

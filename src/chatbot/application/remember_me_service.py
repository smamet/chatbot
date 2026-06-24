from __future__ import annotations

import hashlib
import secrets

from itsdangerous import BadSignature, TimestampSigner

from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
from chatbot.config.settings import Settings
from chatbot.domain.models.user import User

REMEMBER_COOKIE_NAME = "chatbot_remember"


def hash_remember_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class RememberMeService:
    def __init__(
        self,
        repository: SqlAlchemyUserRepository,
        *,
        settings: Settings,
    ) -> None:
        self._repo = repository
        self._settings = settings
        secret = settings.session_secret.strip() or settings.app_secret_key.strip() or "change-me-session"
        self._signer = TimestampSigner(secret, salt="chatbot-remember-me")

    def cookie_max_age_seconds(self) -> int:
        return max(1, self._settings.remember_me_max_age_days) * 24 * 60 * 60

    def issue_token(self, user_id: int) -> str:
        raw = secrets.token_urlsafe(32)
        self._repo.set_remember_token_hash(user_id, hash_remember_token(raw))
        return raw

    def revoke_token(self, user_id: int) -> None:
        self._repo.clear_remember_token_hash(user_id)

    def sign_cookie(self, user_id: int, raw_token: str) -> str:
        payload = f"{user_id}:{raw_token}"
        return self._signer.sign(payload.encode("utf-8")).decode("utf-8")

    def unsign_cookie(self, signed: str) -> tuple[int, str] | None:
        try:
            raw = self._signer.unsign(
                signed.encode("utf-8"),
                max_age=self.cookie_max_age_seconds(),
            ).decode("utf-8")
        except BadSignature:
            return None
        user_id_text, _, token = raw.partition(":")
        if not user_id_text or not token:
            return None
        try:
            return int(user_id_text), token
        except ValueError:
            return None

    def authenticate_cookie(self, signed: str) -> User | None:
        parsed = self.unsign_cookie(signed)
        if parsed is None:
            return None
        user_id, raw_token = parsed
        return self.verify_token(user_id, raw_token)

    def verify_token(self, user_id: int, raw_token: str) -> User | None:
        stored = self._repo.get_remember_token_hash(user_id)
        if not stored:
            return None
        if not secrets.compare_digest(stored, hash_remember_token(raw_token)):
            return None
        user = self._repo.find_by_id(user_id)
        if user is None or not user.active:
            return None
        return user

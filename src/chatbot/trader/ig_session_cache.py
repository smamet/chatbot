"""Shared IG REST session tokens for stream + live workers.

Avoids concurrent ``/session`` logins that can invalidate CST/XST.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chatbot.trader.ig_connector import IgConnector

logger = logging.getLogger(__name__)

# IG sessions typically last hours; refresh proactively before relying on stale tokens.
_DEFAULT_TTL_SECONDS = 6 * 3600


@dataclass
class CachedIgSession:
    cst: str
    xst: str
    lightstreamer_endpoint: str
    account_id: str
    obtained_at: float

    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.obtained_at)

    def is_fresh(self, *, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> bool:
        return bool(self.cst and self.xst) and self.age_seconds() < ttl_seconds


_LOCK = threading.Lock()
_CACHE: dict[str, CachedIgSession] = {}


def session_cache_key(
    *,
    api_key: str,
    username: str,
    account_id: str,
    acc_type: str,
) -> str:
    return "|".join(
        [
            (api_key or "").strip(),
            (username or "").strip().lower(),
            (account_id or "").strip(),
            (acc_type or "DEMO").strip().upper(),
        ]
    )


def get_cached_session(key: str) -> CachedIgSession | None:
    with _LOCK:
        row = _CACHE.get(key)
        if row is None or not row.is_fresh():
            return None
        return row


def store_cached_session(key: str, session: CachedIgSession) -> None:
    with _LOCK:
        _CACHE[key] = session


def invalidate_cached_session(key: str) -> None:
    with _LOCK:
        _CACHE.pop(key, None)


def clear_session_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def apply_cached_session(ig: IgConnector, session: CachedIgSession) -> None:
    """Apply cached tokens onto a connector without calling ``/session``."""
    ig._cst = session.cst
    ig._security = session.xst
    ig.lightstreamer_endpoint = session.lightstreamer_endpoint or None
    ig.current_account_id = session.account_id or None


def capture_session_from_connector(ig: IgConnector) -> CachedIgSession | None:
    cst = (ig._cst or "").strip()
    xst = (ig._security or "").strip()
    if not cst or not xst:
        return None
    return CachedIgSession(
        cst=cst,
        xst=xst,
        lightstreamer_endpoint=(ig.lightstreamer_endpoint or "").strip(),
        account_id=(ig.current_account_id or ig.config.ig_account_id or "").strip(),
        obtained_at=time.time(),
    )


def login_with_shared_cache(
    ig: IgConnector,
    *,
    force: bool = False,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
) -> CachedIgSession:
    """
    Login via REST once per credential key; reuse CST/XST across workers.

    On ``force`` or missing/stale cache, calls ``ig.login()`` and stores tokens.
    """
    key = session_cache_key(
        api_key=ig.config.ig_api_key or "",
        username=ig.config.ig_username or "",
        account_id=ig.config.ig_account_id or "",
        acc_type=ig.config.ig_acc_type or "DEMO",
    )
    if not force:
        cached = get_cached_session(key)
        if cached is not None and cached.is_fresh(ttl_seconds=ttl_seconds):
            apply_cached_session(ig, cached)
            logger.debug("IG session cache hit key=%s…", key[:24])
            return cached

    with _LOCK:
        # Double-check under lock to coalesce parallel logins.
        if not force:
            cached = _CACHE.get(key)
            if cached is not None and cached.is_fresh(ttl_seconds=ttl_seconds):
                apply_cached_session(ig, cached)
                return cached
        ig.login()
        captured = capture_session_from_connector(ig)
        if captured is None:
            raise RuntimeError("IG login produced no CST/XST tokens")
        _CACHE[key] = captured
        logger.info(
            "IG session cache store account=%s ls=%s",
            captured.account_id or "—",
            captured.lightstreamer_endpoint or "—",
        )
        return captured


def session_to_dict(session: CachedIgSession) -> dict[str, Any]:
    return {
        "cst": session.cst,
        "xst": session.xst,
        "lightstreamer_endpoint": session.lightstreamer_endpoint,
        "account_id": session.account_id,
        "obtained_at": session.obtained_at,
        "age_seconds": session.age_seconds(),
    }

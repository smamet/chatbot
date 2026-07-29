"""Add tenants.bot_type and migrate cac40_backtest integrations into trader config.

Revision ID: 026
Revises: 025
"""

from __future__ import annotations

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import inspect, text

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_LEGACY_TYPE = "cac40_backtest"


def _fernet() -> Fernet | None:
    bind = op.get_bind()
    # Prefer APP_SECRET_KEY from the running app env; alembic loads .env via env.py.
    import os

    key = (os.environ.get("APP_SECRET_KEY") or "").strip()
    if not key:
        return None
    return Fernet(key.encode("utf-8"))


def _decrypt_json(blob: str | None, fernet: Fernet | None) -> dict:
    if not blob or not str(blob).strip() or fernet is None:
        return {}
    try:
        raw = fernet.decrypt(str(blob).encode("ascii"))
    except (InvalidToken, ValueError):
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _encrypt_text(plain: str, fernet: Fernet | None) -> str:
    if not plain or fernet is None:
        return ""
    return fernet.encrypt(plain.encode("utf-8")).decode("ascii")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("tenants")}
    if "bot_type" not in cols:
        op.add_column(
            "tenants",
            sa.Column(
                "bot_type",
                sa.String(length=32),
                nullable=False,
                server_default="assistant",
            ),
        )
        op.create_index("ix_tenants_bot_type", "tenants", ["bot_type"])

    fernet = _fernet()
    rows = bind.execute(
        text(
            "SELECT i.id, i.tenant_id, i.config_enc, i.active, t.config_json "
            "FROM integrations i "
            "JOIN tenants t ON t.id = i.tenant_id "
            "WHERE i.type = :itype"
        ),
        {"itype": _LEGACY_TYPE},
    ).mappings().all()

    for row in rows:
        cfg = _decrypt_json(row["config_enc"], fernet)
        try:
            existing = json.loads(row["config_json"] or "{}")
        except json.JSONDecodeError:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        trader = dict(existing.get("trader") or {}) if isinstance(existing.get("trader"), dict) else {}
        if cfg.get("symbol"):
            trader["symbol"] = str(cfg["symbol"]).strip() or trader.get("symbol", "CAC40")
        if cfg.get("epic"):
            trader["epic"] = str(cfg["epic"]).strip() or trader.get("epic", "IX.D.CAC.BMU.IP")
        if cfg.get("fundmanager_url"):
            trader["fundmanager_url"] = str(cfg["fundmanager_url"]).strip()
        token = str(cfg.get("fundmanager_token") or "").strip()
        if token:
            enc = _encrypt_text(token, fernet)
            if enc:
                trader["fundmanager_token_enc"] = enc
            else:
                trader["fundmanager_token"] = token
        if cfg.get("max_open_positions") is not None:
            try:
                trader["max_open_positions"] = int(cfg["max_open_positions"])
            except (TypeError, ValueError):
                pass
        trader.setdefault("market_profile", "cac40")
        existing["trader"] = trader
        # Drop legacy allowlist entry; traders don't use Integrations for markets.
        allowed = existing.get("allowed_integrations")
        if isinstance(allowed, list):
            existing["allowed_integrations"] = [
                x for x in allowed if str(x).strip().lower() != _LEGACY_TYPE
            ]
        # Any tenant that had a CAC40 integration row becomes a trader bot.
        bind.execute(
            text("UPDATE tenants SET config_json = :cj, bot_type = 'trader' WHERE id = :tid"),
            {
                "cj": json.dumps(existing, ensure_ascii=True),
                "tid": row["tenant_id"],
            },
        )

    bind.execute(text("DELETE FROM integrations WHERE type = :itype"), {"itype": _LEGACY_TYPE})


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("tenants")}
    if "bot_type" in cols:
        op.drop_index("ix_tenants_bot_type", table_name="tenants")
        op.drop_column("tenants", "bot_type")

"""Remove trading-bot rows left by CAC40/IG experiments (no DDL).

Revision ID: 026
Revises: 025
"""

from __future__ import annotations

import json
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_ALLOWLIST_KEYS = ("allowed_connectors", "allowed_integrations")


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("DELETE FROM connectors WHERE type = 'ig'"))
    conn.execute(sa.text("DELETE FROM integrations WHERE type = 'cac40_backtest'"))
    conn.execute(sa.text("DELETE FROM api_usage_daily WHERE operation = 'cac40'"))

    rows = conn.execute(sa.text("SELECT id, config_json FROM tenants")).fetchall()
    for tenant_id, config_json in rows:
        raw = config_json or "{}"
        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(cfg, dict):
            continue
        if not any(k in cfg for k in _ALLOWLIST_KEYS):
            continue
        for key in _ALLOWLIST_KEYS:
            cfg.pop(key, None)
        conn.execute(
            sa.text("UPDATE tenants SET config_json = :cfg WHERE id = :id"),
            {"cfg": json.dumps(cfg, separators=(",", ":")), "id": tenant_id},
        )


def downgrade() -> None:
    # Data deletion is intentionally one-way.
    pass

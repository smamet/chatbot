"""Per-tenant client billing rates on tenants.

Revision ID: 016
Revises: 015
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("tenants")}
    if "client_billing_input_per_million_usd" not in cols:
        op.add_column(
            "tenants",
            sa.Column("client_billing_input_per_million_usd", sa.Numeric(10, 4), nullable=True),
        )
    if "client_billing_output_per_million_usd" not in cols:
        op.add_column(
            "tenants",
            sa.Column("client_billing_output_per_million_usd", sa.Numeric(10, 4), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("tenants")}
    if "client_billing_output_per_million_usd" in cols:
        op.drop_column("tenants", "client_billing_output_per_million_usd")
    if "client_billing_input_per_million_usd" in cols:
        op.drop_column("tenants", "client_billing_input_per_million_usd")

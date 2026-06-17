from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from sqlalchemy import BigInteger, Boolean, Date, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TenantRow(Base):
    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    token_hash: Mapped[str] = mapped_column(String(128))
    prompt: Mapped[str] = mapped_column(Text(), default="")
    hook_instructions: Mapped[str | None] = mapped_column(Text(), nullable=True)
    gemini_api_key_enc: Mapped[str | None] = mapped_column(Text(), nullable=True)
    config_json: Mapped[str] = mapped_column(Text(), default="{}")
    active: Mapped[bool] = mapped_column(Boolean(), default=True)
    client_billing_input_per_million_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )
    client_billing_output_per_million_usd: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class MessageRow(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_tenant_session_created", "tenant_id", "session_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(256), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text())
    context_debug_json: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class IngestedFileRow(Base):
    __tablename__ = "ingested_files"
    __table_args__ = (UniqueConstraint("tenant_id", "path", name="uq_ingested_tenant_path"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    path: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class OrderRow(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_tenant_customer_key_status", "tenant_id", "customer_key", "status"),
        Index("ix_orders_tenant_editable_until_status", "tenant_id", "editable_until", "status"),
        Index("ix_orders_tenant_session_id", "tenant_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    customer_key: Mapped[str] = mapped_column(String(128), nullable=False)
    customer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    customer_tel: Mapped[str | None] = mapped_column(String(64), nullable=True)
    delivery_address: Mapped[str | None] = mapped_column(Text(), nullable=True)
    delivery_pin: Mapped[str | None] = mapped_column(Text(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    editable_until: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    items: Mapped[list["OrderItemRow"]] = relationship(
        back_populates="order",
        cascade="all, delete-orphan",
    )


class OrderItemRow(Base):
    __tablename__ = "order_items"
    __table_args__ = (Index("ix_order_items_order_id", "order_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    product: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    order: Mapped["OrderRow"] = relationship(back_populates="items")


class OrderEventRow(Base):
    __tablename__ = "order_events"
    __table_args__ = (
        Index("ix_order_events_tenant_customer_created", "tenant_id", "customer_key", "created_at"),
        Index("ix_order_events_tenant_session_created", "tenant_id", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    customer_key: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    command_json: Mapped[str] = mapped_column(Text(), nullable=False)
    conversation_context: Mapped[str] = mapped_column(Text(), nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(32))
    active: Mapped[bool] = mapped_column(Boolean(), default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class UserBotAccessRow(Base):
    __tablename__ = "user_bot_access"
    __table_args__ = (UniqueConstraint("user_id", "tenant_id", name="uq_user_bot_access"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)


class ConnectorRow(Base):
    __tablename__ = "connectors"
    __table_args__ = (Index("ix_connectors_tenant_dir_type", "tenant_id", "direction", "type"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    type: Mapped[str] = mapped_column(String(32))
    mode: Mapped[str] = mapped_column(String(16), default="direct")
    config_enc: Mapped[str] = mapped_column(Text(), default="")
    active: Mapped[bool] = mapped_column(Boolean(), default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class MailConnectionRow(Base):
    __tablename__ = "mail_connections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    label: Mapped[str] = mapped_column(String(128), default="")
    provider: Mapped[str] = mapped_column(String(32))
    mailbox_email: Mapped[str] = mapped_column(String(255), default="")
    config_enc: Mapped[str] = mapped_column(Text(), default="")
    active: Mapped[bool] = mapped_column(Boolean(), default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class IntegrationRow(Base):
    __tablename__ = "integrations"
    __table_args__ = (Index("ix_integrations_tenant_type", "tenant_id", "type"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    type: Mapped[str] = mapped_column(String(32))
    config_enc: Mapped[str] = mapped_column(Text(), default="")
    active: Mapped[bool] = mapped_column(Boolean(), default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PendingReplyRow(Base):
    __tablename__ = "pending_replies"
    __table_args__ = (Index("ix_pending_replies_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    connector_id: Mapped[int] = mapped_column(ForeignKey("connectors.id"))
    session_id: Mapped[str] = mapped_column(String(256))
    channel: Mapped[str] = mapped_column(String(32))
    recipient_id: Mapped[str] = mapped_column(String(256))
    draft_text: Mapped[str] = mapped_column(Text())
    draft_html: Mapped[str | None] = mapped_column(Text(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    hook_event_id: Mapped[int | None] = mapped_column(ForeignKey("hook_events.id"), nullable=True)
    fulfillment_kind: Mapped[str] = mapped_column(String(32), default="reply_only")
    quote_proposal_json: Mapped[str | None] = mapped_column(Text(), nullable=True)
    quote_resolved_json: Mapped[str | None] = mapped_column(Text(), nullable=True)
    quote_external_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attachments_json: Mapped[str | None] = mapped_column(Text(), nullable=True)
    fulfillment_error: Mapped[str | None] = mapped_column(Text(), nullable=True)
    quote_erp_modified: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PendingReplyEditRow(Base):
    __tablename__ = "pending_reply_edits"
    __table_args__ = (
        Index("ix_pending_reply_edits_tenant_reply", "tenant_id", "pending_reply_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    pending_reply_id: Mapped[int] = mapped_column(ForeignKey("pending_replies.id"))
    edited_by: Mapped[str] = mapped_column(String(256), default="")
    body_before: Mapped[str] = mapped_column(Text())
    body_after: Mapped[str] = mapped_column(Text())
    diff: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PendingReplyAuditEventRow(Base):
    __tablename__ = "pending_reply_audit_events"
    __table_args__ = (
        Index("ix_pending_reply_audit_tenant_reply", "tenant_id", "pending_reply_id"),
        Index("ix_pending_reply_audit_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    pending_reply_id: Mapped[int] = mapped_column(ForeignKey("pending_replies.id"))
    action: Mapped[str] = mapped_column(String(32))
    actor_email: Mapped[str] = mapped_column(String(256), default="")
    detail_json: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class HookEventRow(Base):
    __tablename__ = "hook_events"
    __table_args__ = (
        Index("ix_hook_events_status_id", "status", "id"),
        Index("ix_hook_events_tenant_created", "tenant_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(256))
    type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text())
    status: Mapped[str] = mapped_column(String(32), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MailDraftRow(Base):
    __tablename__ = "mail_drafts"
    __table_args__ = (Index("ix_mail_drafts_tenant_status", "tenant_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    imap_uid: Mapped[str] = mapped_column(String(128), default="")
    from_addr: Mapped[str] = mapped_column(String(512), default="")
    to_addr: Mapped[str] = mapped_column(String(512), default="")
    subject: Mapped[str] = mapped_column(String(1024), default="")
    body_in: Mapped[str] = mapped_column(Text(), default="")
    draft_reply: Mapped[str] = mapped_column(Text(), default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    rating: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MailImapUidRow(Base):
    __tablename__ = "mail_imap_uids"
    __table_args__ = (Index("uq_mail_imap_uids_tenant_imap_uid", "tenant_id", "imap_uid", unique=True),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    imap_uid: Mapped[str] = mapped_column(String(128), default="")
    disposition: Mapped[str] = mapped_column(String(32), default="skipped")
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class TestChatSessionRow(Base):
    __test__ = False

    __tablename__ = "test_chat_sessions"
    __table_args__ = (
        UniqueConstraint("tenant_id", "session_id", name="uq_test_chat_sessions_tenant_session"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(256))
    label: Mapped[str] = mapped_column(String(256))
    last_quote_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ApiUsageDailyRow(Base):
    __tablename__ = "api_usage_daily"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "usage_date",
            "operation",
            "model",
            name="uq_api_usage_daily",
        ),
        Index("ix_api_usage_daily_tenant_date", "tenant_id", "usage_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    usage_date: Mapped[date] = mapped_column(Date(), nullable=False)
    operation: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), default="")
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    call_count: Mapped[int] = mapped_column(Integer, default=0)


class DiskUsageDailyRow(Base):
    __tablename__ = "disk_usage_daily"
    __table_args__ = (
        UniqueConstraint("tenant_id", "snapshot_date", name="uq_disk_usage_daily"),
        Index("ix_disk_usage_daily_tenant_date", "tenant_id", "snapshot_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[int | None] = mapped_column(ForeignKey("tenants.id"), nullable=True, index=True)
    snapshot_date: Mapped[date] = mapped_column(Date(), nullable=False)
    total_bytes: Mapped[int] = mapped_column(BigInteger, default=0)

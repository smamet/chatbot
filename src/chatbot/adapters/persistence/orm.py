from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(256), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class IngestedFileRow(Base):
    __tablename__ = "ingested_files"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(1024), unique=True)
    content_hash: Mapped[str] = mapped_column(String(128))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class OrderRow(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("ix_orders_customer_key_status", "customer_key", "status"),
        Index("ix_orders_editable_until_status", "editable_until", "status"),
        Index("ix_orders_session_id", "session_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
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
        Index("ix_order_events_customer_key_created_at", "customer_key", "created_at"),
        Index("ix_order_events_session_id_created_at", "session_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False)
    customer_key: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    command_json: Mapped[str] = mapped_column(Text(), nullable=False)
    conversation_context: Mapped[str] = mapped_column(Text(), nullable=False)
    error_detail: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

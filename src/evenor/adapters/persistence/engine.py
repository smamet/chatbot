from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from evenor.adapters.persistence.orm import Base
from evenor.config.settings import Settings


def create_db_engine(settings: Settings, *, for_tests: bool = False) -> Engine:
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url, future=True)
    if for_tests:
        Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine):
    return sessionmaker(engine, expire_on_commit=False, future=True)

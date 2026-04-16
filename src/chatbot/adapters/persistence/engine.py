from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from chatbot.adapters.persistence.orm import Base
from chatbot.config.settings import Settings


def create_db_engine(settings: Settings) -> Engine:
    if settings.database_url.startswith("sqlite:///"):
        db_path = Path(settings.database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url, future=True)
    Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine):
    return sessionmaker(engine, expire_on_commit=False, future=True)

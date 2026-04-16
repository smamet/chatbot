from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.sync_service import IngestSyncService
from chatbot.config.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Customer chatbot CLI.")


@app.command("sync")
def sync_cmd(path: Annotated[Path, typer.Argument(exists=False, help="File or directory to reconcile with the index")]) -> None:
    """Prune missing files under this root from the index, then (re)ingest supported documents."""
    settings = get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    embedder = GeminiEmbedder()
    store = LanceVectorStore(settings.lancedb_path)
    with factory() as session:
        svc = IngestSyncService(settings=settings, embedder=embedder, vector_store=store, session=session)
        for line in svc.reconcile_root(path):
            typer.echo(line)
        session.commit()


@app.command("version")
def version_cmd() -> None:
    """Print package version string."""
    typer.echo("chatbot 0.1.0")


if __name__ == "__main__":
    app()

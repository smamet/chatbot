from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.ingest_service import IngestService
from chatbot.config.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Customer chatbot CLI.")


@app.command("ingest")
def ingest_cmd(path: Annotated[Path, typer.Argument(exists=False, help="File or directory to ingest")]) -> None:
    """Index supported documents (pdf, csv, xlsx, xls) into the vector store."""
    settings = get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    embedder = GeminiEmbedder(api_key=settings.gemini_api_key, model=settings.embedding_model)
    store = LanceVectorStore(settings.lancedb_path)
    with factory() as session:
        svc = IngestService(settings=settings, embedder=embedder, vector_store=store, session=session)
        for line in svc.ingest_path(path):
            typer.echo(line)
        session.commit()


@app.command("version")
def version_cmd() -> None:
    """Print package version string."""
    typer.echo("chatbot 0.1.0")


if __name__ == "__main__":
    app()

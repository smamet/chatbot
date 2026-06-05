from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir, tenant_lancedb_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.config.settings import get_settings
from chatbot.domain.models.tenant import TenantConfig

app = typer.Typer(no_args_is_help=True, help="Multi-tenant chatbot CLI.")


def _resolve_password(password: str | None, *, command_hint: str) -> str:
    import sys

    if password:
        return password
    if sys.stdin.isatty():
        return typer.prompt("Password", hide_input=True)
    typer.echo(
        f"No TTY: pass --password/-p, e.g.\n  chatbot {command_hint} -p 'your-password'",
        err=True,
    )
    raise typer.Exit(1)


@app.command("sync")
def sync_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    path: Annotated[
        Path | None,
        typer.Argument(help="File or directory (default: tenant docs folder)"),
    ] = None,
    fresh: Annotated[
        bool,
        typer.Option("--fresh", help="Clear tenant RAG index before ingesting"),
    ] = False,
) -> None:
    """Reconcile documents for a tenant with its LanceDB index."""
    settings = get_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    embedder = GeminiEmbedder()
    with factory() as session:
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        tenant = tenant_svc.get_by_slug(slug)
        if tenant is None:
            typer.echo(f"Unknown tenant slug: {slug}", err=True)
            raise typer.Exit(1)
        root = path or tenant_docs_dir(settings, slug)
        merged = merge_tenant_settings(settings, tenant)
        store = LanceVectorStore(tenant_lancedb_dir(settings, slug))
        svc = IngestSyncService(
            settings=merged,
            embedder=embedder,
            vector_store=store,
            session=session,
            tenant_id=tenant.id,
        )
        for line in svc.reconcile_root(root, fresh=fresh):
            typer.echo(line)
        session.commit()


@app.command("user-create")
def user_create_cmd(
    email: Annotated[str, typer.Argument()],
    password: Annotated[
        str | None,
        typer.Option("--password", "-p", help="Password (use when stdin is not a TTY, e.g. ./sail)"),
    ] = None,
    role: Annotated[str, typer.Option("--role")] = "admin",
) -> None:
    from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
    from chatbot.application.user_service import UserService
    from chatbot.domain.models.user import UserRole

    resolved = _resolve_password(
        password, command_hint="user-create admin@example.com --role admin"
    )

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = UserService(SqlAlchemyUserRepository(session))
        if svc.find_by_email(email):
            typer.echo(
                f"User already exists: {email.lower().strip()}\n"
                "Log in at /auth/login, or reset password:\n"
                f"  ./sail chatbot user-set-password {email} -p 'new-password'",
                err=True,
            )
            raise typer.Exit(1)
        user = svc.create_user(email=email, password=resolved, role=UserRole(role))
        session.commit()
        typer.echo(f"user_id={user.id} email={user.email} role={user.role}")


@app.command("user-set-password")
def user_set_password_cmd(
    email: Annotated[str, typer.Argument()],
    password: Annotated[str | None, typer.Option("--password", "-p")] = None,
) -> None:
    from chatbot.adapters.persistence.user_repository import SqlAlchemyUserRepository
    from chatbot.application.user_service import UserService

    resolved = _resolve_password(password, command_hint=f"user-set-password {email}")

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = UserService(SqlAlchemyUserRepository(session))
        user = svc.set_password(email, resolved)
        if user is None:
            typer.echo(f"No user with email: {email}", err=True)
            raise typer.Exit(1)
        session.commit()
        typer.echo(f"Password updated for {user.email}")


@app.command("tenant-create")
def tenant_create_cmd(
    name: Annotated[str, typer.Argument(help="Display name")],
    slug: Annotated[str | None, typer.Option("--slug", help="URL slug (auto if omitted)")] = None,
) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = TenantService(SqlAlchemyTenantRepository(session))
        result = svc.create_tenant(name=name, slug=slug)
        session.commit()
        typer.echo(f"slug={result.tenant.slug}")
        typer.echo(f"token={result.token}")


@app.command("serve")
def serve_cmd(
    host: Annotated[str, typer.Option("--host", "-h")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    import uvicorn

    uvicorn.run("chatbot.interfaces.api.main:app", host=host, port=port, reload=reload)


@app.command("version")
def version_cmd() -> None:
    typer.echo("chatbot 0.2.0")


if __name__ == "__main__":
    app()

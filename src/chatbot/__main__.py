from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from chatbot.adapters.embeddings.gemini_embedder import GeminiEmbedder
from chatbot.adapters.persistence.engine import create_db_engine, session_factory
from chatbot.adapters.persistence.tenant_paths import tenant_catalog_dir, tenant_docs_dir, tenant_lancedb_dir
from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
from chatbot.adapters.rag.lance_vector_store import LanceVectorStore
from chatbot.application.erpnext_catalog_sync_service import (
    catalog_rag_index_plan,
    reconcile_catalog_rag,
    sync_erpnext_catalog_for_tenant,
)
from chatbot.application.sync_service import IngestSyncService
from chatbot.application.tenant_flush_service import TenantFlushError, TenantFlushService
from chatbot.application.tenant_service import TenantService
from chatbot.application.tenant_settings import merge_tenant_settings
from chatbot.config.settings import get_settings
from chatbot.domain.models.tenant import TenantConfig

app = typer.Typer(no_args_is_help=True, help="Multi-tenant chatbot CLI.")
catalog_rag_app = typer.Typer(no_args_is_help=True, help="Catalog RAG index (shared with dashboard/worker).")
app.add_typer(catalog_rag_app, name="catalog-rag")


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


def _catalog_rag_on_file_done(progress, task_id):
    def callback(path: Path, log_line: str) -> None:
        progress.advance(task_id)
        progress.console.print(f"  {path.name}: {log_line}")

    return callback


def _catalog_rag_progress(**kwargs):
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        **kwargs,
    )


@catalog_rag_app.command("rebuild")
def catalog_rag_rebuild_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    all_files: Annotated[
        bool,
        typer.Option("--all", help="Scan all catalog files (unchanged still skip embedding)"),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show index plan only")] = False,
) -> None:
    """Resume or rebuild catalog RAG index (missing/changed files only by default)."""
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        tenant = tenant_svc.get_by_slug(slug)
        if tenant is None:
            typer.echo(f"Unknown tenant slug: {slug}", err=True)
            raise typer.Exit(1)

        catalog_root = tenant_catalog_dir(settings, slug)
        plan = catalog_rag_index_plan(session, tenant.id, catalog_root)
        total = len(list(catalog_root.glob("*.md"))) if all_files else len(plan.needs_embed)

        typer.echo(
            f"Catalog: {len(plan.needs_embed) + len(plan.already_indexed)} files, "
            f"{len(plan.needs_embed)} need embedding, "
            f"{len(plan.already_indexed)} already indexed"
        )
        if dry_run:
            typer.echo(f"Dry run: would process {total} file(s)")
            return

        if total == 0:
            typer.echo("Nothing to embed.")
            return

        with _catalog_rag_progress() as progress:
            task_id = progress.add_task("Embedding catalog", total=total)
            logs = reconcile_catalog_rag(
                session,
                settings=settings,
                tenant_id=tenant.id,
                slug=slug,
                paths_to_reindex=plan.needs_embed if not all_files else None,
                reconcile_all=all_files,
                on_file_done=_catalog_rag_on_file_done(progress, task_id),
                commit_each_batch=True,
            )
        session.commit()

    for line in logs:
        if line.startswith(("ingested ", "unchanged:")):
            continue
        typer.echo(line)


@catalog_rag_app.command("sync")
def catalog_rag_sync_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    all_files: Annotated[
        bool,
        typer.Option("--all", help="Force full catalog RAG reconcile after ERP sync"),
    ] = False,
) -> None:
    """Fetch ERPNext catalog and reconcile RAG (same as dashboard Sync catalog now)."""
    from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
    from chatbot.application.integration_service import IntegrationService
    from chatbot.domain.models.integration import IntegrationType

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
        tenant = tenant_svc.get_by_slug(slug)
        if tenant is None:
            typer.echo(f"Unknown tenant slug: {slug}", err=True)
            raise typer.Exit(1)

        integration = IntegrationService(SqlAlchemyIntegrationRepository(session)).find_active(
            tenant.id,
            type=IntegrationType.ERPNEXT,
        )
        if integration is None:
            typer.echo(f"No active ERPNext integration for {slug}", err=True)
            raise typer.Exit(1)

        catalog_root = tenant_catalog_dir(settings, slug)
        plan = catalog_rag_index_plan(session, tenant.id, catalog_root)
        embed_total = len(list(catalog_root.glob("*.md"))) if all_files else len(plan.needs_embed)

        with _catalog_rag_progress(disable=embed_total == 0) as progress:
            task_id = progress.add_task("Embedding catalog", total=max(embed_total, 1))
            result = sync_erpnext_catalog_for_tenant(
                session,
                settings=settings,
                tenant_id=tenant.id,
                tenant_slug=slug,
                config=integration.config,
                force_rag_reconcile=all_files,
                on_file_done=_catalog_rag_on_file_done(progress, task_id),
                commit_each_batch=True,
            )
        session.commit()

    typer.echo(result.message)
    if not result.ok:
        raise typer.Exit(1)
    for line in result.logs:
        if line.startswith(("ingested ", "unchanged:", "wrote:", "removed:")):
            continue
        typer.echo(line)


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


@app.command("bot-flush")
def bot_flush_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation (required without a TTY)"),
    ] = False,
    no_backup: Annotated[
        bool,
        typer.Option("--no-backup", help="Do not save a backup before flushing"),
    ] = False,
) -> None:
    """Clear all chats and operational logs for a bot; keep RAG, connectors, and config."""
    import sys

    if not yes:
        if not sys.stdin.isatty():
            typer.echo(
                "No TTY: pass --yes / -y to confirm, e.g.\n  chatbot bot-flush my-bot --yes",
                err=True,
            )
            raise typer.Exit(1)
        typed = typer.prompt(f"Type the slug to confirm flush of '{slug}'")
        if typed.strip() != slug:
            typer.echo("Slug did not match; aborted.", err=True)
            raise typer.Exit(1)

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = TenantFlushService(session, settings=settings)
        try:
            logs, backup_path = svc.flush(slug, backup=not no_backup)
        except TenantFlushError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        session.commit()
        for line in logs:
            typer.echo(line)
    if backup_path is not None:
        typer.echo(f"Restore with: chatbot bot-restore {slug} {backup_path} --yes")
    typer.echo("Done.")


@app.command("bot-restore")
def bot_restore_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    backup: Annotated[Path, typer.Argument(help="Backup directory from bot-flush")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation (required without a TTY)"),
    ] = False,
) -> None:
    """Restore operational data from a bot-flush backup."""
    import sys

    if not yes:
        if not sys.stdin.isatty():
            typer.echo(
                "No TTY: pass --yes / -y to confirm, e.g.\n"
                f"  chatbot bot-restore {slug} {backup} --yes",
                err=True,
            )
            raise typer.Exit(1)
        typed = typer.prompt(f"Type the slug to confirm restore of '{slug}'")
        if typed.strip() != slug:
            typer.echo("Slug did not match; aborted.", err=True)
            raise typer.Exit(1)

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = TenantFlushService(session, settings=settings)
        try:
            logs = svc.restore(slug, backup)
        except TenantFlushError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        session.commit()
        for line in logs:
            typer.echo(line)
    typer.echo("Done.")


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

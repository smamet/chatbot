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
    files_only: Annotated[
        bool,
        typer.Option("--files-only", help="Fetch ERPNext catalog markdown only; skip RAG ingest"),
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
        embed_total = 0 if files_only else (
            len(list(catalog_root.glob("*.md"))) if all_files else len(plan.needs_embed)
        )

        with _catalog_rag_progress(disable=embed_total == 0) as progress:
            task_id = progress.add_task("Embedding catalog", total=max(embed_total, 1))
            result = sync_erpnext_catalog_for_tenant(
                session,
                settings=settings,
                tenant_id=tenant.id,
                tenant_slug=slug,
                config=integration.config,
                force_rag_reconcile=all_files,
                skip_rag_ingest=files_only,
                on_file_done=_catalog_rag_on_file_done(progress, task_id) if not files_only else None,
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
    bot_type: Annotated[
        str,
        typer.Option("--type", help="assistant | trader"),
    ] = "assistant",
) -> None:
    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        svc = TenantService(SqlAlchemyTenantRepository(session))
        result = svc.create_tenant(name=name, slug=slug, bot_type=bot_type)
        session.commit()
        typer.echo(f"slug={result.tenant.slug}")
        typer.echo(f"bot_type={result.tenant.bot_type.value}")
        typer.echo(f"token={result.token}")


@app.command("bot-duplicate")
def bot_duplicate_cmd(
    source: Annotated[str, typer.Argument(help="Source tenant slug")],
    name: Annotated[str, typer.Option("--name", "-n", help="Display name for the clone")],
    slug: Annotated[str | None, typer.Option("--slug", help="URL slug (auto if omitted)")] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Trader market profile override (e.g. eurusd)"),
    ] = None,
    symbol: Annotated[
        str | None,
        typer.Option("--symbol", help="Trader symbol override"),
    ] = None,
    epic: Annotated[
        str | None,
        typer.Option("--epic", help="IG epic override"),
    ] = None,
    reset_prompt: Annotated[
        bool,
        typer.Option(
            "--reset-prompt",
            help="Replace system prompt with the market profile default",
        ),
    ] = False,
) -> None:
    """Clone credentials and settings into a new bot (no RAG / operational data)."""
    from chatbot.application.tenant_duplicate_service import (
        TenantDuplicateError,
        duplicate_tenant,
    )

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        try:
            result = duplicate_tenant(
                session,
                settings,
                source,
                name=name,
                slug=slug,
                market_profile=profile,
                symbol=symbol,
                epic=epic,
                reset_prompt_from_profile=reset_prompt,
            )
        except TenantDuplicateError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        session.commit()
        typer.echo(f"source={source}")
        typer.echo(f"slug={result.tenant.slug}")
        typer.echo(f"bot_type={result.tenant.bot_type.value}")
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
    keep_monitoring: Annotated[
        bool,
        typer.Option(
            "--keep-monitoring",
            help="Keep api_usage_daily and disk_usage_daily rows for this bot",
        ),
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
            logs, backup_path = svc.flush(
                slug,
                backup=not no_backup,
                keep_monitoring=keep_monitoring,
            )
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


@app.command("mail-inbox-preview")
def mail_inbox_preview_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Number of messages")] = 5,
) -> None:
    """List recent IMAP messages and whether the mail worker would process them."""
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.application.mail_inbox_preview_service import preview_tenant_inbox

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_slug(slug)
        if tenant is None:
            typer.echo(f"Unknown tenant slug: {slug}", err=True)
            raise typer.Exit(1)
        result = preview_tenant_inbox(session, tenant_id=tenant.id, settings=settings, limit=limit)
    if not result.ok:
        typer.echo(result.message, err=True)
        if result.error:
            typer.echo(result.error, err=True)
        raise typer.Exit(1)
    typer.echo(result.message)
    if result.mailbox:
        typer.echo(f"Mailbox: {result.mailbox}")
    if result.process_since_display and result.process_since_display != "—":
        typer.echo(f"Process since (server display): {result.process_since_display}")
    if not result.messages:
        typer.echo("(INBOX empty or no parseable messages)")
    for index, msg in enumerate(result.messages, start=1):
        typer.echo(
            f"{index}. UID {msg.uid} | {msg.received_at or 'no date'} | from {msg.from_addr}"
        )
        typer.echo(f"   Subject: {msg.subject}")
        if msg.eligible:
            typer.echo("   ELIGIBLE")
        else:
            typer.echo(f"   SKIP: {msg.skip_reason}")


@app.command("mail-connection-migrate")
def mail_connection_migrate_cmd(
    slug: Annotated[str, typer.Argument(help="Tenant slug")],
) -> None:
    """Migrate per-connector OAuth email configs to shared mail connections."""
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.application.mail_connection_migrate_service import MailConnectionMigrateService

    settings = get_settings()
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    with factory() as session:
        tenant = SqlAlchemyTenantRepository(session).find_by_slug(slug)
        if tenant is None:
            typer.echo(f"Unknown tenant slug: {slug}", err=True)
            raise typer.Exit(1)
        result = MailConnectionMigrateService(session).migrate_tenant(tenant.id)
        session.commit()
        typer.echo(result.message)
        typer.echo(
            f"connections_created={result.connections_created} "
            f"connectors_updated={result.connectors_updated}"
        )


trader_app = typer.Typer(no_args_is_help=True, help="Trader bot / backtest / live.")
app.add_typer(trader_app, name="trader")


@trader_app.command("backtest")
def trader_backtest_cmd(
    ohlc: Annotated[Path, typer.Argument(help="15m OHLCV CSV path")],
    out: Annotated[Path, typer.Option("--out", help="Run output directory")] = Path("data/trader/cli_runs"),
    max_open_positions: Annotated[int, typer.Option("--max-open-positions")] = 4,
    llm_mode: Annotated[str, typer.Option("--llm-mode")] = "replay",
    llm_every_bars: Annotated[int, typer.Option("--llm-every-bars")] = 4,
    spread_points: Annotated[float, typer.Option("--spread-points")] = 1.5,
) -> None:
    """Run a hedge-mode trader backtest (HedgeLedger)."""
    from chatbot.trader.backtest_engine import BacktestEngine, new_run_dir
    from chatbot.trader.config import TraderConfig

    settings = get_settings()
    cfg = TraderConfig(
        max_open_positions=max_open_positions,
        llm_mode=llm_mode,
        llm_every_bars=llm_every_bars,
        spread_points=spread_points,
    )
    run_dir = new_run_dir(out)
    engine = BacktestEngine(
        cfg,
        ohlc_path=ohlc,
        run_dir=run_dir,
        api_key=settings.gemini_api_key or "",
    )
    report = engine.run()
    typer.echo(f"run_dir={run_dir}")
    typer.echo(
        f"equity={report['final_equity']:.2f} dd={report['max_drawdown']:.2f} "
        f"trades={report['trades']} winrate={report['winrate']:.0%}"
    )


@trader_app.command("live")
def trader_live_cmd(
    config_json: Annotated[
        Path | None,
        typer.Option("--config", help="JSON config file (TraderConfig fields)"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run")] = True,
    once: Annotated[bool, typer.Option("--once", help="Single cycle then exit")] = False,
    sleep_seconds: Annotated[int, typer.Option("--sleep-seconds")] = 900,
) -> None:
    """Run the 15m live/demo loop (fail-closed)."""
    import json as _json

    from chatbot.trader.config import TraderConfig
    from chatbot.trader.scheduler import LiveScheduler

    settings = get_settings()
    data = _json.loads(config_json.read_text()) if config_json else {}
    cfg = TraderConfig.from_dict(data)
    journal = settings.data_root / "trader" / "live"
    sched = LiveScheduler(
        cfg,
        api_key=settings.gemini_api_key or "",
        journal_dir=journal,
        dry_run=dry_run,
        sleep_seconds=sleep_seconds,
    )
    if once:
        payload = sched.run_once()
        typer.echo(_json.dumps(payload, indent=2, default=str))
    else:
        sched.run_forever()


@trader_app.command("stream-probe")
def trader_stream_probe_cmd(
    from_db_slug: Annotated[
        str | None,
        typer.Option("--from-db-slug", help="Load IG connector credentials from this bot slug"),
    ] = None,
    epics: Annotated[
        str,
        typer.Option("--epics", help="Comma-separated epics"),
    ] = "IX.D.CAC.BMU.IP,CS.D.EURUSD.MINI.IP",
    seconds: Annotated[float, typer.Option("--seconds", help="How long to listen")] = 45.0,
) -> None:
    """DEMO Lightstreamer probe: PRICE ticks + TRADE + tick→15m HLC buckets."""
    import json as _json

    from chatbot.trader.ig_stream_probe import run_ig_stream_probe

    cfg = _ig_config_from_cli(from_db_slug)
    epic_list = [e.strip() for e in epics.split(",") if e.strip()]
    result = run_ig_stream_probe(cfg, epics=epic_list, seconds=seconds)
    typer.echo(_json.dumps(result.to_dict(), indent=2, default=str))
    raise typer.Exit(0 if result.ok else 1)


@trader_app.command("order-probe")
def trader_order_probe_cmd(
    from_db_slug: Annotated[
        str | None,
        typer.Option("--from-db-slug", help="Load IG connector credentials from this bot slug"),
    ] = None,
    allow_market_orders: Annotated[
        bool,
        typer.Option(
            "--allow-market-orders/--no-allow-market-orders",
            help="Also open/close a tiny DEMO market position (off by default)",
        ),
    ] = False,
) -> None:
    """DEMO France CAC LIMIT/STOP ±TP matrix (+ EURUSD smoke); market only with flag."""
    from chatbot.application.connector_test_service import run_ig_stream_order_probe

    cfg = _ig_config_from_cli(from_db_slug)
    result = run_ig_stream_order_probe(cfg, allow_market_orders=allow_market_orders)
    typer.echo(result.message)
    raise typer.Exit(0 if result.ok else 1)


def _ig_config_from_cli(from_db_slug: str | None) -> dict:
    """Load DEMO IG config from a bot slug's stored connector."""
    if not from_db_slug:
        raise typer.BadParameter("Provide --from-db-slug with a bot that has an IG connector")
    from sqlalchemy import select

    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.adapters.persistence.engine import create_db_engine, session_factory
    from chatbot.adapters.persistence.orm import TenantRow
    from chatbot.application.connector_service import ConnectorService
    from chatbot.config.settings import get_settings

    Factory = session_factory(create_db_engine(get_settings()))
    with Factory() as session:
        tenant = session.execute(
            select(TenantRow).where(TenantRow.slug == from_db_slug.strip())
        ).scalar_one_or_none()
        if tenant is None:
            raise typer.BadParameter(f"Unknown bot slug: {from_db_slug}")
        cfg = ConnectorService(SqlAlchemyConnectorRepository(session)).get_ig_config(tenant.id)
    if not cfg:
        raise typer.BadParameter(f"No IG connector on bot {from_db_slug}")
    out = dict(cfg)
    out["acc_type"] = "DEMO"
    return out


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

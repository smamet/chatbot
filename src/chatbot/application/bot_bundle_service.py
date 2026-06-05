from __future__ import annotations

import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sqlalchemy.orm import Session

from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
from chatbot.adapters.persistence.tenant_paths import tenant_docs_dir
from chatbot.application.connector_service import ConnectorService
from chatbot.application.tenant_service import TenantService
from chatbot.config.settings import Settings
from chatbot.domain.models.connector import ConnectorDirection, ConnectorMode, ConnectorType
from chatbot.domain.models.tenant import Tenant, TenantConfig, TenantCreateResult

EXPORT_VERSION = 1
MANIFEST_NAME = "manifest.json"
DOCUMENTS_PREFIX = "documents/"


class ImportMode(StrEnum):
    CREATE = "create"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class ImportBundleResult:
    tenant: Tenant
    token: str | None = None


class BotBundleError(ValueError):
    pass


def _config_dict(cfg: TenantConfig) -> dict:
    return json.loads(cfg.to_json())


def _config_from_dict(data: dict) -> TenantConfig:
    return TenantConfig.from_json(json.dumps(data))


def _connector_to_dict(conn) -> dict:
    return {
        "direction": conn.direction.value,
        "type": conn.type.value,
        "mode": conn.mode.value,
        "active": conn.active,
        "config": conn.config,
    }


def build_manifest(tenant: Tenant, connectors: list) -> dict:
    return {
        "export_version": EXPORT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "source_slug": tenant.slug,
        "bot": {
            "name": tenant.name,
            "active": tenant.active,
            "prompt": tenant.prompt,
            "hook_instructions": tenant.hook_instructions,
            "gemini_api_key": tenant.gemini_api_key,
            "config": _config_dict(tenant.config),
        },
        "connectors": [_connector_to_dict(c) for c in connectors],
    }


def build_export(
    tenant: Tenant,
    settings: Settings,
    session: Session,
) -> bytes:
    connectors = ConnectorService(SqlAlchemyConnectorRepository(session)).list_for_tenant(
        tenant.id
    )
    manifest = build_manifest(tenant, connectors)
    docs_root = tenant_docs_dir(settings, tenant.slug)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        if docs_root.is_dir():
            for path in docs_root.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(docs_root).as_posix()
                zf.write(path, f"{DOCUMENTS_PREFIX}{rel}")
    return buffer.getvalue()


def _parse_manifest(zf: zipfile.ZipFile) -> dict:
    try:
        raw = zf.read(MANIFEST_NAME).decode("utf-8")
        data = json.loads(raw)
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BotBundleError("Invalid bundle: missing or malformed manifest.json") from exc
    if not isinstance(data, dict):
        raise BotBundleError("Invalid bundle: manifest must be a JSON object")
    if data.get("export_version") != EXPORT_VERSION:
        raise BotBundleError(f"Unsupported export version: {data.get('export_version')}")
    bot = data.get("bot")
    if not isinstance(bot, dict):
        raise BotBundleError("Invalid bundle: bot section missing")
    return data


def _clear_docs_dir(docs_dir: Path) -> None:
    if docs_dir.exists():
        shutil.rmtree(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)


def _extract_documents(zf: zipfile.ZipFile, docs_dir: Path) -> None:
    _clear_docs_dir(docs_dir)
    prefix_len = len(DOCUMENTS_PREFIX)
    for name in zf.namelist():
        if not name.startswith(DOCUMENTS_PREFIX) or name.endswith("/"):
            continue
        rel = name[prefix_len:]
        if not rel or ".." in Path(rel).parts:
            raise BotBundleError(f"Invalid document path in bundle: {rel}")
        dest = (docs_dir / rel).resolve()
        if not str(dest).startswith(str(docs_dir.resolve())):
            raise BotBundleError(f"Invalid document path in bundle: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))


def _replace_connectors(
    connector_service: ConnectorService,
    tenant_id: int,
    connectors_data: list,
) -> None:
    for existing in connector_service.list_for_tenant(tenant_id):
        connector_service.delete(existing.id)
    if not isinstance(connectors_data, list):
        return
    for raw in connectors_data:
        if not isinstance(raw, dict):
            continue
        try:
            direction = ConnectorDirection(str(raw["direction"]))
            ctype = ConnectorType(str(raw["type"]))
            mode = ConnectorMode(str(raw.get("mode", ConnectorMode.DIRECT.value)))
        except (KeyError, ValueError):
            continue
        config = raw.get("config")
        if not isinstance(config, dict):
            config = {}
        connector_service.upsert(
            tenant_id=tenant_id,
            direction=direction,
            type=ctype,
            mode=mode,
            config=config,
            active=bool(raw.get("active", True)),
        )


def _apply_bot_fields(
    tenant_service: TenantService,
    tenant_id: int,
    bot: dict,
    *,
    name: str | None = None,
) -> Tenant | None:
    cfg_raw = bot.get("config")
    config = _config_from_dict(cfg_raw) if isinstance(cfg_raw, dict) else TenantConfig()
    gemini_key = bot.get("gemini_api_key")
    if isinstance(gemini_key, str):
        gemini_key = gemini_key.strip() or None
    else:
        gemini_key = None
    return tenant_service.update_tenant(
        tenant_id,
        name=name or str(bot.get("name", "Imported bot")).strip(),
        prompt=str(bot.get("prompt", "")),
        config=config,
        active=bool(bot.get("active", True)),
        hook_instructions=bot.get("hook_instructions"),
        update_hook_instructions=True,
        gemini_api_key=gemini_key,
        update_gemini_api_key=True,
    )


def import_bundle(
    zip_bytes: bytes,
    *,
    mode: ImportMode,
    settings: Settings,
    session: Session,
    tenant_service: TenantService,
    target_slug: str | None = None,
    new_name: str | None = None,
) -> ImportBundleResult:
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise BotBundleError("Invalid ZIP file") from exc

    with zf:
        manifest = _parse_manifest(zf)
        bot = manifest["bot"]
        connectors_data = manifest.get("connectors", [])

        connector_service = ConnectorService(SqlAlchemyConnectorRepository(session))

        if mode == ImportMode.CREATE:
            cfg_raw = bot.get("config")
            config = _config_from_dict(cfg_raw) if isinstance(cfg_raw, dict) else TenantConfig()
            gemini_key = bot.get("gemini_api_key")
            if isinstance(gemini_key, str):
                gemini_key = gemini_key.strip() or None
            else:
                gemini_key = None
            create_name = (new_name or str(bot.get("name", "Imported bot"))).strip()
            result: TenantCreateResult = tenant_service.create_tenant(
                name=create_name,
                prompt=str(bot.get("prompt", "You are a helpful assistant.")),
                config=config,
                hook_instructions=bot.get("hook_instructions"),
                gemini_api_key=gemini_key,
            )
            tenant = tenant_service.update_tenant(
                result.tenant.id,
                active=bool(bot.get("active", True)),
            ) or result.tenant
            _replace_connectors(connector_service, tenant.id, connectors_data)
            _extract_documents(zf, tenant_docs_dir(settings, tenant.slug))
            return ImportBundleResult(tenant=tenant, token=result.token)

        if mode == ImportMode.OVERWRITE:
            if not target_slug:
                raise BotBundleError("Target bot slug is required for overwrite")
            tenant = tenant_service.get_by_slug(target_slug)
            if tenant is None:
                raise BotBundleError(f"Target bot not found: {target_slug}")
            updated = _apply_bot_fields(
                tenant_service,
                tenant.id,
                bot,
                name=new_name.strip() if new_name and new_name.strip() else None,
            )
            if updated is None:
                raise BotBundleError("Failed to update target bot")
            _replace_connectors(connector_service, tenant.id, connectors_data)
            _extract_documents(zf, tenant_docs_dir(settings, tenant.slug))
            return ImportBundleResult(tenant=updated)

        raise BotBundleError(f"Unknown import mode: {mode}")

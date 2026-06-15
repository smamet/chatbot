# Agent guide — chatbot platform

## Architecture

- **Hexagonal layout**: `domain/` (models + contracts), `application/` (use cases), `adapters/` (SQLAlchemy, LanceDB, Gemini), `interfaces/` (FastAPI, web dashboard, workers).
- **Multi-tenant**: every row scoped by `tenant_id`; LanceDB under `data/lancedb/{slug}/`.
- **RAG sources**: uploaded docs in `data/docs/{slug}/`; ERPNext catalogue snapshots in `data/catalog/{slug}/` (one `.md` per item, isolated from document sync).
- **Auth**: `POST /c/{slug}/chat` with `Authorization: Bearer <tenant_token>` (token must match slug). Admin API: `ADMIN_TOKEN`. Dashboard: session cookie (`/auth/login`).
- **Hooks**: LLM appends global marker `===HOOK===` + JSON → `hook_events` → `worker-automation` dispatches **automation modules** (`core.orders` → local DB; `erpnext.quote` → validation queue + ERPNext on approve). Module list per bot in `config_json.automation_modules`. Run `./sail migrate` after pull for `006_mail_drafts_uid_unique`.
- **ERPNext catalog → RAG**: `worker-catalog` polls tenants with active ERPNext + `sync_catalog_to_rag`; fetches paginated Item + Bin (stock aggregate) + Item Price (`catalog_price_list`, default Standard Selling), writes `data/catalog/{slug}/*.md`, then `reconcile_catalog_rag()` → `IngestSyncService.ingest_paths_batched()` on that folder only (never `data/docs/`). Price line: Item Price → standard_rate → `not available`. Config in integration schema (`sync_catalog_to_rag`, `catalog_sync_interval_minutes`, `catalog_include_stock`, `catalog_price_list`); runtime metadata in encrypted config (`catalog_last_sync_at`, `catalog_last_item_count`, `catalog_last_error`) — merge-safe, not in schema fields. Manual sync: dashboard **Sync catalog now** or `./sail chatbot catalog-rag sync {slug}`. Resume partial RAG index: `./sail chatbot catalog-rag rebuild {slug}` (uses `catalog_rag_index_plan()` for missing/changed files). Embedding retries (429/503) live in `GeminiEmbedder` only — shared by UI, workers, CLI, document sync.
- **Connectors**: per-tenant channel creds in `connectors.config_enc` (Fernet via `APP_SECRET_KEY`).

## Docker (Sail)

```bash
./sail up -d
./sail chatbot user-create admin@example.com -p 'your-password' --role admin
./sail chatbot user-set-password admin@example.com -p 'new-password'  # if user already exists
./sail chatbot tenant-create "Name" --slug my-client
./sail chatbot sync my-client /app/data/docs
./sail test
./sail shell
```

Compose services: `db` (MySQL), `api`, `worker-automation`, `worker-mail`, `worker-catalog`, `caddy`. Root `.env` is loaded via `env_file`; `DATABASE_URL` is overridden to MySQL inside containers.

**Dev:** `docker-compose.override.yml` bind-mounts `./src` — edit Python without `./sail build`. It also bind-mounts `./data/docs` and `./data/catalog` over the `app_data` volume for those paths (LanceDB stays in the volume). API runs with `--reload`. Rebuild only after dependency/Dockerfile changes. Restart workers after code changes: `./sail restart worker-automation` / `./sail restart worker-mail` / `./sail restart worker-catalog`.

**Email dev (GreenMail + Mailpit):** `./sail up -d --profile dev` starts GreenMail (IMAP 3143, inject SMTP 3025) and Mailpit (outbound SMTP 1025, UI http://127.0.0.1:8025). IN connector → GreenMail; OUT connector → Mailpit. Test email inject uses GreenMail SMTP (not OUT). See [docs/dev/greenmail.md](docs/dev/greenmail.md). Dashboard **Test email** tab requires `DEV_MODE=true`.

## Common tasks

### Add a tenant

1. Dashboard **Bots** → open bot, or `./sail chatbot tenant-create "Name" --slug my-client`
2. Save the token shown once.
3. Upload docs (dashboard or admin API), run **Sync** (documents only — `data/docs/{slug}/`).
4. Configure **Connectors** (WhatsApp/Meta) for webhook URLs `/webhooks/{channel}/{slug}`.
5. Optional: **Integrations → ERPNext** — enable **Sync catalog to RAG** for live catalogue in RAG (`data/catalog/{slug}/`).

### ERPNext catalog sync

- Service: `src/chatbot/application/erpnext_catalog_sync_service.py`; client pagination in `ErpNextClient.list_catalog_items` / `fetch_stock_totals` / `fetch_price_list_rates`.
- Worker: `python -m chatbot.interfaces.worker_catalog` (`--once` for a single poll). Poll interval: `CATALOG_POLL_SECONDS`; per-bot interval: `catalog_sync_interval_minutes` in integration config.
- `catalog_price_list`: absent → `Standard Selling`; empty string → Item Price skipped (standard_rate only). Never emit price `0.0` in markdown.
- Do not call document `reconcile_root` on `data/catalog/` or vice versa — separate roots, separate `IngestSyncService` runs.
- Metadata keys (`catalog_last_*`) are outside the integration schema; `_merge_integration_config` preserves them on dashboard save. Worker re-reads config before writing metadata.
- **CLI (same pipeline as dashboard/worker):**
  ```bash
  ./sail chatbot catalog-rag rebuild {slug}           # resume: embed missing/changed only
  ./sail chatbot catalog-rag rebuild {slug} --dry-run
  ./sail chatbot catalog-rag rebuild {slug} --all     # force full catalog scan (unchanged skip embed)
  ./sail chatbot catalog-rag sync {slug}              # ERPNext fetch + RAG reconcile
  ```
- `catalog_rag_index_plan()` → `needs_embed` / `already_indexed` (content hash vs `ingested_files`). `reconcile_catalog_rag(..., on_file_done, commit_each_batch)` — auto `commit_each_batch` when >50 files.
- **Embedding:** `GeminiEmbedder.embed_texts()` — no fixed RPM throttle; retry 429 (`Retry-After` or 30×2ⁿ s, max 120) and 503 (5×2ⁿ s, max 30). Env: `EMBED_RETRY_MAX`, `EMBED_RETRY_BASE_429_SECONDS`, `EMBED_RETRY_BASE_503_SECONDS`. SDK HTTP retry disabled (`attempts=1`); app loop owns backoff.

### Run locally (no Docker)

```bash
pip install -e ".[dev]"
mkdir -p data
uvicorn chatbot.interfaces.api.main:app --reload
python -m chatbot.interfaces.worker_automation
python -m chatbot.interfaces.worker_mail
python -m chatbot.interfaces.worker_catalog
```

Dashboard: http://127.0.0.1:8000/auth/login

### Migrations (MySQL / prod)

```bash
alembic upgrade head
# or ./sail migrate
```

Tests use SQLite + `create_db_engine(..., for_tests=True)` (no Alembic).

If Docker MySQL migrations failed mid-run (`Table 'tenants' already exists`, etc.), reset the DB volume and re-run:

```bash
docker compose down -v
./sail up -d
```

(`-v` drops `mysql_data`; `app_data` LanceDB/docs persist unless you remove that volume too.)

### Tests

```bash
pytest
# ./sail test
```

Key tests: `test_tenant_isolation.py`, `test_api_chat.py`, `test_dashboard_web.py`, `test_hooks_flow.py`, `test_hook_extractor.py`, `test_mail_worker.py`, `test_imap_client.py`, `test_erpnext_catalog_sync_service.py`, `test_erpnext_client.py`, `test_gemini_embedder.py`, `test_cli_catalog_rag.py`.

## Environment (`.env`)

**Required:** `GEMINI_API_KEY`, `ADMIN_TOKEN`, `APP_SECRET_KEY`, `SESSION_SECRET`.

**Storage:** `DATA_ROOT`, `LANCEDB_ROOT`; `DATABASE_URL` (SQLite local, MySQL in Docker).

**Workers:** `HOOK_POLL_SECONDS`, `MAIL_POLL_SECONDS`, `CATALOG_POLL_SECONDS` (catalog worker poll; default 300).

**Embedding retries:** `EMBED_RETRY_MAX` (default 5), `EMBED_RETRY_BASE_429_SECONDS` (30), `EMBED_RETRY_BASE_503_SECONDS` (5).

**Not used anymore:** `CHAT_API_SECRET`, `PROMPT_PATH`, `LANCEDB_PATH`, `WEBHOOK_TENANT_SLUG`.

**Channel secrets:** dashboard Connectors, not `.env` (optional env fallbacks in `settings.py` only for migration).

## Do not

- Query `messages` / `orders` / `hook_events` without `tenant_id` filter.
- Share LanceDB directories between tenants.
- Store tenant tokens in plain text (only `token_hash` in DB).
- Put order/WhatsApp logic in `ChatService` — use hooks + automation worker.
- Run full-catalog RAG reconcile on `data/docs/` when syncing ERPNext items (use `data/catalog/{slug}/` only).
- Assume RAG stock/price is real-time — it reflects the last catalog sync snapshot.

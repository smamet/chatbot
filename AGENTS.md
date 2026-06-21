# Agent guide — chatbot platform

## Architecture

- **Hexagonal layout**: `domain/` (models + contracts), `application/` (use cases), `adapters/` (SQLAlchemy, LanceDB, Gemini), `interfaces/` (FastAPI, web dashboard, workers).
- **Multi-tenant**: every row scoped by `tenant_id`; LanceDB under `data/lancedb/{slug}/`.
- **RAG sources**: uploaded docs in `data/docs/{slug}/`; ERPNext catalogue snapshots in `data/catalog/{slug}/` (one `.md` per item, isolated from document sync).
- **Auth**: `POST /c/{slug}/chat` with `Authorization: Bearer <tenant_token>` (token must match slug). Admin API: `ADMIN_TOKEN`. Dashboard: session cookie (`/auth/login`). Dashboard roles — see **Dashboard roles** below.
- **Hooks**: LLM appends global marker `===HOOK===` + JSON → `hook_events` → `worker-automation` dispatches **automation modules** (`core.orders` → local DB; `erpnext.quote` → validation queue + ERPNext on approve). Module list per bot in `config_json.automation_modules`. Run `./sail migrate` after pull for `013_validation_audit` (validation audit + `client_operator` role rename).
- **ERPNext catalog → RAG**: `worker-catalog` polls tenants with active ERPNext + `sync_catalog_to_rag`; fetches paginated Item + Bin (stock aggregate) + Item Price (`catalog_price_list`, default Standard Selling), writes `data/catalog/{slug}/*.md`, then `reconcile_catalog_rag()` → `IngestSyncService.ingest_paths_batched()` on that folder only (never `data/docs/`). Price line: Item Price → standard_rate → `not available`. Config in integration schema (`sync_catalog_to_rag`, `catalog_sync_interval_minutes`, `catalog_include_stock`, `catalog_price_list`); runtime metadata in encrypted config (`catalog_last_sync_at`, `catalog_last_item_count`, `catalog_last_error`) — merge-safe, not in schema fields. Manual sync: dashboard **Sync catalog now** or `./sail chatbot catalog-rag sync {slug}`. Resume partial RAG index: `./sail chatbot catalog-rag rebuild {slug}` (uses `catalog_rag_index_plan()` for missing/changed files). Embedding retries (429/503) live in `GeminiEmbedder` only — shared by UI, workers, CLI, document sync.
- **Connectors**: per-tenant channel creds in `connectors.config_enc` (Fernet via `APP_SECRET_KEY`). Email connectors support **password**, **Microsoft OAuth**, or **Google OAuth** (`auth_type`). OAuth mailboxes use reusable **`mail_connections`** rows (one Connect covers IMAP + SMTP); email IN/OUT connectors reference `mail_connection_id` in config. Platform OAuth app credentials optional in `.env` (`MICROSOFT_MAIL_CLIENT_ID`/`SECRET`, `GOOGLE_MAIL_CLIENT_ID`/`SECRET`) — override per-connection client ID/secret for enterprise tenants. Refresh tokens live on the **connection** row. Single OAuth redirect URI for the platform: `{PUBLIC_BASE_URL}/dashboard/mail-oauth/callback` (register once in Entra/Google; legacy per-connection callback path still works). Password/GreenMail and Mailjet/Mailgun stay inline on the connector. Dashboard **Connectors** tab: **Mail connections** panel (create, Connect, Test IMAP/SMTP) + connector form with connection dropdown. Gmail SMTP: port 587 STARTTLS only (not 465). Mailbox owner must complete OAuth consent. Migrate legacy per-connector OAuth: `./sail chatbot mail-connection-migrate {slug}`. Run `./sail migrate` after pull for `017_mail_connections`.
- **Monitoring**: Gemini calls wrapped with `MeteredLlmClient` / `MeteredEmbedder` → daily rollups in `api_usage_daily` (`UsageRecorderService`). Live disk scan via `DiskUsageService`; nightly history in `disk_usage_daily` (`DiskSnapshotService`, triggered from `worker-catalog` once per UTC day). Dashboard: admin `/dashboard/monitoring` + per-bot `?tab=monitoring`. Cost estimates from local list prices (`gemini_pricing.py`, default model `gemini-2.5-flash` at $0.30/$2.50 per 1M in/out) — Google has no per-model pricing API. Two-tier billing: **internal** (per-model Google rates, admin only) vs **client billable** (flat $/M from `CLIENT_BILLING_*` env + optional per-tenant override on `tenants.client_billing_*`, admin-only — not in `config_json`). Run `./sail migrate` after pull for `014_api_usage_daily`, `015_disk_usage_daily`, `016_tenant_client_billing`.
- **Email threading**: `email_threads` table (per-sender thread, `thread_key` = 12-char hex). New inbound mail → `session_id` = `email:{addr}~{thread_key}`; legacy `email:{addr}` sessions are not migrated. `EmailThreadResolver`: RFC headers (`Message-ID` / `In-Reply-To` / `References`) → normalized subject → LLM (`REWRITE_MODEL`, operation `email_thread`) when ambiguous and `EMAIL_THREAD_LLM_ENABLED=true`. `EmailReplyParser` extracts `body_new` (quoted text stripped) — only `body_new` goes to the chat LLM; raw body stays in `mail_drafts.body_in`. Resolution audit persisted on `mail_drafts.thread_resolution_json`; validation inbox shows green/yellow ● (tooltip = method; yellow = LLM called). On validation approve, outbound SMTP/Mailjet/Mailgun includes `In-Reply-To` + `References`; `outbound_email_messages` records sent `Message-ID`. **Regenerate from raw** (email pending, non-quote): `POST …/validation/{id}/regenerate` re-runs LLM with full history + `body_in` as last user turn; updates draft + messages; audit `regenerated`. **Blocked senders**: `TenantConfig.email_blocked_senders` (Config tab); listener skips before thread/LLM (`mail_imap_uids.disposition=blacklisted`); operators **Reject & Blacklist** on validation detail; Validation sub-tab **Blacklist** lists addresses (+ Unblock for editors). Run `./sail migrate` after pull for `019_email_threads`, `020_mail_draft_thread_resolution`, `021_resanitize_mail_draft_body_new` (one-time backfill of `body_new` from `body_in`), `022_backfill_html_body_new` (HTML→plain backfill for existing drafts).

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

Compose services: `db` (MySQL), `api`, `worker-automation`, `worker-mail`, `worker-catalog`; optional `caddy` (`./sail up -d --profile caddy`). App data bind-mounts `./data` → `/app/data` by default (docs, catalog, LanceDB, attachments). Root `.env` is loaded via `env_file`; `DATABASE_URL` is overridden to MySQL inside containers.

**Dev:** `docker-compose.override.yml` bind-mounts `./src` — edit Python without `./sail build`. API runs with `--reload`. Rebuild only after dependency/Dockerfile changes. Restart workers after code changes: `./sail restart worker-automation` / `./sail restart worker-mail` / `./sail restart worker-catalog`.

**Email dev (GreenMail + Mailpit):** `./sail up -d --profile dev` starts GreenMail (IMAP 3143, inject SMTP 3025) and Mailpit (outbound SMTP 1025, UI http://127.0.0.1:8025). IN connector → GreenMail; OUT connector → Mailpit. Test email inject uses GreenMail SMTP (not OUT). See [docs/dev/greenmail.md](docs/dev/greenmail.md). Dashboard **Test email** tab requires `DEV_MODE=true`.

## Common tasks

### Dashboard roles

| Role | Scope | Dashboard |
|------|-------|-----------|
| `admin` | All bots | Users, hooks, monitoring (global + all bots), create/delete bots, all tabs |
| `client_admin` | Assigned bots (`user_bot_access`) | Config, connectors, integrations, documents, validation, **monitoring** (assigned bots — tokens/disk charts; **client billable cost only**) |
| `client_operator` | Assigned bots | **Validation only** — inbox, detail, history; full operator actions (edit, approve/reject, attachments) |

- **Login home** (`client_operator`): one assigned bot → validation inbox; two or more → bot picker (**Open** → validation); none → empty list.
- **Permissions**: `UserService.can_edit` (bot config) vs `can_validate` (validation mutations). Operators have `can_validate` but not `can_edit`. **Test chat**: `UserService.can_use_full_test_chat` — admin only (identity email/phone, channel simulation, multi-session sidebar). `client_admin` gets anonymous test chat only (cookie-persisted `test:` session, messages loaded on tab open, minimal UI).
- **CLI**: `./sail chatbot user-create … --role client_operator` (also `admin`, `client_admin`). Assign bots on dashboard **Users** → user detail (admin only).

### Add a tenant

1. Dashboard **Bots** → open bot, or `./sail chatbot tenant-create "Name" --slug my-client`
2. Save the token shown once.
3. Upload docs (dashboard or admin API), run **Sync** (documents only — `data/docs/{slug}/`).
4. Configure **Connectors** (WhatsApp/Meta) for webhook URLs `/webhooks/{channel}/{slug}`.
5. Optional: **Integrations → ERPNext** — enable **Sync catalog to RAG** for live catalogue in RAG (`data/catalog/{slug}/`).

### Flush bot operational data (keep RAG)

Service: `src/chatbot/application/tenant_flush_service.py` (`TenantFlushService.flush`).

```bash
./sail chatbot bot-flush {slug} --yes
./sail chatbot bot-restore {slug} data/backups/{slug}/{timestamp} --yes
```

Clears messages, `hook_events`, validation queue (`pending_replies` + edits + audit), orders, `mail_drafts`, `email_threads`, `outbound_email_messages`, `test_chat_sessions`, `mail_imap_uids`, tenant monitoring rows (`api_usage_daily`, `disk_usage_daily` for that bot only — not host snapshots), and runtime dirs `data/attachments/{slug}/`, `data/quotes/{slug}/`. **Does not** delete the tenant, connectors, integrations, `ingested_files`, LanceDB, `data/docs/` or `data/catalog/`. Token unchanged. Destructive — requires `--yes` without a TTY or slug confirmation on a TTY.

**Backup (default on flush):** `data/backups/{slug}/{timestamp}/` with `manifest.json`, `operational.json`, and copies of attachment/quote dirs. `--no-backup` skips. `--keep-monitoring` skips `api_usage_daily` / tenant `disk_usage_daily` deletion. Monitoring is not restored from backup. `bot-restore` replaces current operational data with the backup (same slug + `tenant_id` only).

**Test chat:** Admin — identity email/phone, channel simulation, multi-session sidebar (`test_chat_sessions`, shows dashboard user per session). `client_admin` — one anonymous `test:` session per browser (HttpOnly cookie per bot, not shared between users); no sidebar/identity/channel UI. Clear history only works for anonymous `test:` sessions (403 for `email:`/`whatsapp:`). Production-only conversations appear under **History** only.

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

### Validation queue (email attachments)

- Inbox: dashboard **Validation** tab (`?tab=validation`) — sub-tabs **Pending** / **Approved** / **Rejected** / **Blacklist**; channel icon column (no Kind/Channel text); **Activity** feed at bottom. **Open** → `/dashboard/bots/{slug}/validation/{reply_id}`.
- **Audit**: draft edits → `pending_reply_edits` (diff, `edited_by`); approve/reject/regenerate/reject-blacklist/attachments → `pending_reply_audit_events`; terminal replies store `resolved_by` / `resolved_at` on `pending_replies`. Service: `validation_audit_service.py`.
- **Message bubbles** (clean/raw + token footer + **Show raw** toggle): validation detail, **History** tab, **Test chat** — shared helpers in `validation_message_ui.py`, macro `message_bubble.html`, JS `message-bubbles.js`.
- Intended operator role: `client_operator`.
- Email replies: WYSIWYG draft + drag-and-drop attachments on the detail page (`pending_replies.attachments_json`).
- User uploads: `data/attachments/{slug}/{reply_id}/` via `store_outbound_attachment()` in `quote_pdf_storage.py`.
- Quote PDFs: `data/quotes/{slug}/` (ERPNext); merged with manual attachments on approve.
- Cleanup: `delete_attachment_files()` on approve/reject (no TTL timer).
- Limits: `ATTACHMENT_MAX_BYTES` (default 10 MiB), `ATTACHMENT_MAX_TOTAL_BYTES` (default 25 MiB).
- API: `POST/DELETE /dashboard/bots/{slug}/validation/{reply_id}/attachments` (email + pending only).

### Monitoring (API usage, disk, estimated cost)

- **Global (admin):** `/dashboard/monitoring` — platform token chart (30 days), disk charts (sum of tenants + host), per-bot table with internal + client billable columns.
- **Per-bot:** `?tab=monitoring` (`can_edit` required). Token in/out totals, daily charts (Chart.js), live disk breakdown, daily usage table with est. cost. Admins also see internal estimate + Google pricing link + **Client billing rates** form (`POST /dashboard/bots/{slug}/monitoring/client-billing`).
- **Metering:** `MeteredLlmClient` / `MeteredEmbedder` at Gemini choke points; `recorder=None` for CLI paths that should not bill. Recorder errors are swallowed so chat/sync never fails on usage DB issues.
- **Tables:** `api_usage_daily` (per tenant/date/operation/model); `disk_usage_daily` (`tenant_id` NULL = host snapshot).
- **Disk snapshots:** `DiskSnapshotService.record_all_if_due()` in `worker_catalog.run_once()` after catalog sync — runs even when no ERPNext tenants; idempotent upsert for today (UTC). Env `DISK_SNAPSHOT_ENABLED` (default on). Restart `worker-catalog` after deploy.
- **Cost:** `UsageCostService` + `gemini_pricing.py`. Override list prices via `GEMINI_PRICING_JSON` (JSON map of model → `{input, output}` per 1M USD). Client view uses flat `CLIENT_BILLING_INPUT_PER_MILLION_USD` / `CLIENT_BILLING_OUTPUT_PER_MILLION_USD` or per-tenant nullable columns on `tenants` (admin sets on monitoring tab).
- **Display:** `format_count()` / `format_usd()` Jinja filters; large token counts use thousands separators.

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

(`-v` drops `mysql_data`; `./data` on the host is unchanged unless you delete it.)

### Tests

```bash
pytest
# ./sail test
```

Key tests: `test_tenant_isolation.py`, `test_api_chat.py`, `test_dashboard_web.py`, `test_hooks_flow.py`, `test_hook_extractor.py`, `test_mail_worker.py`, `test_imap_client.py`, `test_mail_connection_service.py`, `test_mail_connection_migrate_service.py`, `test_mail_oauth_service.py`, `test_erpnext_catalog_sync_service.py`, `test_erpnext_client.py`, `test_gemini_embedder.py`, `test_cli_catalog_rag.py`, `test_tenant_flush_service.py`, `test_usage_cost_service.py`, `test_monitoring_format.py`, `test_disk_snapshot_service.py`, `test_email_subject.py`, `test_email_reply_parser.py`, `test_email_session_id.py`, `test_email_thread_resolver.py`, `test_email_thread_disambiguator.py`, `test_email_thread_resolution.py`, `test_email_thread_settings.py`, `test_email_body_sanitize.py`, `test_mail_listener_threads.py`.

## Environment (`.env`)

**Required:** `GEMINI_API_KEY`, `ADMIN_TOKEN`, `APP_SECRET_KEY`, `SESSION_SECRET`.

**Storage:** `DATA_ROOT`, `LANCEDB_ROOT`; `DATABASE_URL` (SQLite local, MySQL in Docker).

**Workers:** `HOOK_POLL_SECONDS`, `MAIL_POLL_SECONDS`, `CATALOG_POLL_SECONDS` (catalog worker poll; default 300). Catalog worker also runs daily disk snapshots when due.

**Models (defaults):** `CHAT_MODEL`, `REWRITE_MODEL` (default `gemini-2.5-flash`), `EMBEDDING_MODEL` (default `gemini-embedding-001`).

**Monitoring / billing estimates:** `GEMINI_PRICING_JSON` (optional JSON override of per-model list prices), `CLIENT_BILLING_INPUT_PER_MILLION_USD` / `CLIENT_BILLING_OUTPUT_PER_MILLION_USD` (flat client-facing $/M; default 1.0 / 3.0), `DISK_SNAPSHOT_ENABLED` (default true).

**Embedding retries:** `EMBED_RETRY_MAX` (default 5), `EMBED_RETRY_BASE_429_SECONDS` (30), `EMBED_RETRY_BASE_503_SECONDS` (5).

**Validation email attachments:** `ATTACHMENT_MAX_BYTES` (default 10 MiB), `ATTACHMENT_MAX_TOTAL_BYTES` (default 25 MiB).

**Email threading:** `EMAIL_THREAD_STALE_DAYS` (default 90), `EMAIL_THREAD_SUBJECT_SIMILARITY` (default 0.85), `EMAIL_THREAD_LLM_ENABLED` (default false — LLM classify runs only when ambiguous and flag enabled), `EMAIL_THREAD_LLM_MIN_CONFIDENCE` (default 0.7).

**Mail OAuth (optional platform app):** `MICROSOFT_MAIL_CLIENT_ID` / `MICROSOFT_MAIL_CLIENT_SECRET`, `GOOGLE_MAIL_CLIENT_ID` / `GOOGLE_MAIL_CLIENT_SECRET`. Register redirect URI `{PUBLIC_BASE_URL}/dashboard/mail-oauth/callback` once in Entra/Google. Per-connection client ID/secret on the dashboard overrides when env is unset.

**Not used anymore:** `CHAT_API_SECRET`, `PROMPT_PATH`, `LANCEDB_PATH`, `WEBHOOK_TENANT_SLUG`.

**Channel secrets:** dashboard Connectors, not `.env` (optional env fallbacks in `settings.py` only for migration).

## Do not

- Query `messages` / `orders` / `hook_events` without `tenant_id` filter.
- Share LanceDB directories between tenants.
- Store tenant tokens in plain text (only `token_hash` in DB).
- Put order/WhatsApp logic in `ChatService` — use hooks + automation worker.
- Run full-catalog RAG reconcile on `data/docs/` when syncing ERPNext items (use `data/catalog/{slug}/` only).
- Assume RAG stock/price is real-time — it reflects the last catalog sync snapshot.
- Use `bot-flush` when you need a clean operational slate — it does not rebuild or clear RAG (`ingested_files`, LanceDB, `data/docs/`, `data/catalog/`).
- Store client billable rates in `config_json` — use `tenants.client_billing_*` columns (admin-only); `client_admin` can edit bot config.
- Pass raw inbound email body (`mail_drafts.body_in`) to the chat LLM when `body_new` is available — `body_in` is for audit/validation only.
- Mix legacy `email:{addr}` and threaded `email:{addr}~{thread_key}` session IDs in one `list_messages` query — they are separate conversation scopes.
- Query `email_threads` without `tenant_id` filter.
- Treat monitoring cost figures as list-price estimates — not Google invoices; refresh rates via `GEMINI_PRICING_JSON` when Google changes pricing.

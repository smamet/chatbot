# Agent guide — chatbot platform

## Architecture

- **Hexagonal layout**: `domain/` (models + contracts), `application/` (use cases), `adapters/` (SQLAlchemy, LanceDB, Gemini), `interfaces/` (FastAPI, web dashboard, workers).
- **Multi-tenant**: every row scoped by `tenant_id`; LanceDB under `data/lancedb/{slug}/`.
- **Auth**: `POST /c/{slug}/chat` with `Authorization: Bearer <tenant_token>` (token must match slug). Admin API: `ADMIN_TOKEN`. Dashboard: session cookie (`/auth/login`).
- **Hooks**: LLM appends global marker `===HOOK===` + JSON → `hook_events` → `worker-automation` dispatches **automation modules** (`core.orders` → local DB; `erpnext.quote` → validation queue + ERPNext on approve). Module list per bot in `config_json.automation_modules`. Run `./sail migrate` after pull for `005_pending_reply_quotes`.
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

Compose services: `db` (MySQL), `api`, `worker-automation`, `caddy`. Root `.env` is loaded via `env_file`; `DATABASE_URL` is overridden to MySQL inside containers.

**Dev:** `docker-compose.override.yml` bind-mounts `./src` — edit Python without `./sail build`. API runs with `--reload`. Rebuild only after dependency/Dockerfile changes. Restart worker after automation code changes: `./sail restart worker-automation`.

## Common tasks

### Add a tenant

1. Dashboard **Bots** → open bot, or `./sail chatbot tenant-create "Name" --slug my-client`
2. Save the token shown once.
3. Upload docs (dashboard or admin API), run **Sync**.
4. Configure **Connectors** (WhatsApp/Meta) for webhook URLs `/webhooks/{channel}/{slug}`.

### Run locally (no Docker)

```bash
pip install -e ".[dev]"
mkdir -p data
uvicorn chatbot.interfaces.api.main:app --reload
python -m chatbot.interfaces.worker_automation
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

Key tests: `test_tenant_isolation.py`, `test_api_chat.py`, `test_dashboard_web.py`, `test_hooks_flow.py`, `test_hook_extractor.py`.

## Environment (`.env`)

**Required:** `GEMINI_API_KEY`, `ADMIN_TOKEN`, `APP_SECRET_KEY`, `SESSION_SECRET`.

**Storage:** `DATA_ROOT`, `LANCEDB_ROOT`; `DATABASE_URL` (SQLite local, MySQL in Docker).

**Not used anymore:** `CHAT_API_SECRET`, `PROMPT_PATH`, `LANCEDB_PATH`, `WEBHOOK_TENANT_SLUG`.

**Channel secrets:** dashboard Connectors, not `.env` (optional env fallbacks in `settings.py` only for migration).

## Do not

- Query `messages` / `orders` / `hook_events` without `tenant_id` filter.
- Share LanceDB directories between tenants.
- Store tenant tokens in plain text (only `token_hash` in DB).
- Put order/WhatsApp logic in `ChatService` — use hooks + automation worker.

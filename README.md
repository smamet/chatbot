# Multi-tenant chatbot platform

Python 3.12, FastAPI + Gemini, per-tenant RAG (LanceDB), **HTMX dashboard**, generic **hooks** + automation worker, optional Meta webhooks. Each bot has its own **slug**, **token**, **prompt**, **connectors**, and **vector index**.

## Quick start (Docker + Sail)

[Sail](https://laravel.com/docs/sail)-style helper: **`./sail`** wraps Docker Compose.

```bash
cp .env.example .env
# Set GEMINI_API_KEY, ADMIN_TOKEN, APP_SECRET_KEY (see below)
./sail up -d
./sail chatbot user-create admin@example.com -p 'your-password' --role admin
./sail chatbot tenant-create "My Bot" --slug my-bot
# Save the tenant token printed once → TENANT_SLUG / TENANT_TOKEN in .env for Streamlit dev
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/healthz | Health check |
| http://localhost:8000/auth/login | Dashboard (session login) |
| http://localhost:8000/dashboard/monitoring | API usage & disk monitoring (admin) |
| http://localhost:80/… | Same via Caddy (optional) |
| `POST /c/{slug}/chat` | Chat API (`Authorization: Bearer <tenant_token>`) |

## Dashboard users & roles

| Role | Who | Access |
|------|-----|--------|
| `admin` | Platform admin | Everything — global monitoring, internal cost estimates, per-bot client billing rates |
| `client_admin` | Bot manager | Assigned bots — config, connectors, validation, **monitoring** (tokens, disk, **client billable cost only**) |
| `client_operator` | Validation operator | Assigned bots — validation inbox/detail/history only |

Create an operator: `./sail chatbot user-create op@example.com -p '…' --role client_operator`, then an admin assigns bot access under **Users**. On login, a single-bot operator lands on the validation inbox; multiple bots show a picker where **Open** goes straight to validation.

**Sail commands**

```bash
./sail up -d          # start db, api, workers (data in ./data on host)
./sail up -d --profile caddy   # optional TLS reverse proxy (skip if host nginx/Caddy)
./sail down           # stop stack
./sail logs           # follow logs
./sail shell          # bash in API container
./sail mysql          # MySQL CLI
./sail chatbot …      # CLI (tenant-create, sync, catalog-rag, bot-flush, bot-restore, user-create, …)
./sail migrate        # alembic upgrade head
./sail test           # pytest in container
./sail worker-logs    # automation worker logs
./sail logs worker-catalog   # ERPNext catalog → RAG worker
```

Run any Compose command: `./sail ps`, `./sail restart api`, etc.

**No rebuild for code edits:** `docker-compose.override.yml` mounts `./src` into the API and workers; uvicorn runs with `--reload`. Rebuild only when `pyproject.toml` or the Dockerfile change: `./sail build api`.

**Data on disk:** Compose bind-mounts `./data` (docs, catalog, LanceDB, attachments, backups). Create it once: `mkdir -p data`.

**Migrate from old Docker `app_data` volume** (one-time, if upgrading an existing server):

```bash
mkdir -p ./data
docker compose run --rm -v chatbot_app_data:/from:ro -v "$(pwd)/data":/to alpine \
  sh -c "cp -a /from/. /to/"
./sail up -d
```

(Volume name may be `chatbot_app_data` or `<project>_app_data` — check with `docker volume ls`.)

**First-time secrets in `.env`**

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM + embeddings (fallback if tenant has no override) |
| `CHAT_MODEL` / `REWRITE_MODEL` | Default Gemini chat model (`gemini-2.5-flash`) |
| `ADMIN_TOKEN` | `/admin/*` API + Streamlit legacy admin pages |
| `APP_SECRET_KEY` | Fernet — encrypts connector configs in DB |
| `SESSION_SECRET` | Dashboard session cookies |

Generate Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Automation worker** processes `hook_events` (orders, etc.). Compose starts `worker-automation` automatically; locally: `python -m chatbot.interfaces.worker_automation`.

**Mail worker** polls IMAP inboxes (`worker-mail`). **Catalog worker** syncs ERPNext item snapshots into RAG when enabled per bot (`worker-catalog`) and records **daily disk usage snapshots** (per tenant + host) once per UTC day. **CAC40 OHLC worker** tops up 15m bars from active IG connectors (`worker-cac40-ohlc`, every `CAC40_OHLC_POLL_SECONDS`, default 900). Locally: `python -m chatbot.interfaces.worker_mail` / `python -m chatbot.interfaces.worker_catalog --once` / `python -m chatbot.interfaces.worker_cac40_ohlc --once`.

**Email testing (dev)** — GreenMail (inbound IMAP) + Mailpit (outbound SMTP UI):

```bash
./sail up -d --profile dev
# Set DEV_MODE=true in .env for the dashboard Test email tab
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8025 | Mailpit — view outbound emails after Validation approve |
| http://127.0.0.1:8081 | GreenMail REST API |
| Dashboard → Test email | Inject simulated client mail into bot inbox |

**Validation (email):** inbox on the bot **Validation** tab; open a pending reply to edit the draft, attach files (drag-and-drop), and approve. Attachments are stored under `data/attachments/{slug}/{reply_id}/` and removed on approve/reject. Quote PDFs from ERPNext are attached automatically when applicable.

**Email threading:** each inbound email is assigned to a conversation thread (`thread_key`) using RFC headers (`Message-ID`, `In-Reply-To`, `References`) when present, otherwise normalized subject matching, with an optional LLM disambiguation step for ambiguous cases only (`EMAIL_THREAD_LLM_ENABLED`, default false). New threads use `session_id` = `email:{addr}~{thread_key}` (legacy `email:{addr}` sessions are unchanged). Quoted reply text is stripped before the chat LLM sees the message; the full raw body is kept for validation. On approve, outbound email includes `In-Reply-To` and `References` so the recipient's mail client groups the reply correctly. Env: `EMAIL_THREAD_STALE_DAYS`, `EMAIL_THREAD_LLM_ENABLED`. Run `./sail migrate` for migrations `019_email_threads` and `020_mail_draft_thread_resolution`.

See [docs/dev/greenmail.md](docs/dev/greenmail.md) for connector presets (IN: GreenMail, OUT: Mailpit).

**Monitoring:** admins open **Monitoring** in the top nav for a platform-wide view; bot managers open a bot → **Monitoring** tab. Shows last 30 days of Gemini token usage (in/out), live disk breakdown, trend charts, and **estimated USD cost**. Internal estimates use published Google list prices (`gemini-2.5-flash`: $0.30 / $2.50 per 1M input/output tokens). Client admins see billable amounts only, using platform defaults (`CLIENT_BILLING_*` in `.env`) or per-bot rates set by an admin on the monitoring tab. Override list prices without a deploy via `GEMINI_PRICING_JSON`. Estimates are not invoices — see [Google pricing](https://ai.google.dev/gemini-api/docs/pricing). After upgrading, run `./sail migrate` (migrations `014`–`016`) and `./sail restart worker-catalog` so disk history charts populate.

**Channel credentials** (WhatsApp, Messenger, Instagram) belong in the **dashboard → bot → Connectors**, not in `.env`. Webhook URLs are per bot: `https://<host>/webhooks/whatsapp/{slug}`, etc.

---

## Quick start (local dev, no Docker)

```bash
pyenv install 3.12.8
pyenv virtualenv 3.12.8 chatbot && pyenv local chatbot
pip install -e ".[dev]"
cp .env.example .env
mkdir -p data
uvicorn chatbot.interfaces.api.main:app --reload
```

- Dashboard: http://127.0.0.1:8000/auth/login — `chatbot user-create admin@example.com`
- Workers (separate terminals): `python -m chatbot.interfaces.worker_automation`, `worker_mail`, `worker_catalog`, `worker_cac40_ohlc`
- Migrations (MySQL only): `alembic upgrade head` — tests use SQLite `create_all`, no Alembic

**`.env` hot reload:** most vars reload when `.env` is saved (mtime). **`DATABASE_URL`** still needs an API restart.

---

## Test chat

Use the dashboard **Test chat** tab on a bot detail page (server-side `ChatService`). For API integration tests, call `POST /c/{slug}/chat` with the tenant Bearer token.

---

## Reset bot data (keep RAG)

Wipe all chats and operational logs for a bot **without** recreating it — same slug, token, prompt, connectors, integrations, and full RAG index (`data/docs/`, `data/catalog/`, `data/lancedb/{slug}/`).

```bash
./sail chatbot bot-flush my-bot --yes
# Local (no Docker):
chatbot bot-flush my-bot --yes
```

Without `--yes`, an interactive terminal prompts you to type the slug to confirm. In non-TTY environments (e.g. `./sail`), `--yes` is required.

By default, a backup is saved under `data/backups/{slug}/{timestamp}/` (JSON + `attachments/` and `quotes/` copies). Pass `--no-backup` to skip. After flush, the CLI prints a restore command.

**Rollback:**

```bash
./sail chatbot bot-restore my-bot data/backups/my-bot/20260615T143022Z --yes
```

**Removed:** messages, hook events, validation queue (replies + edits + audit), orders, mail drafts, email threads, outbound email messages, test chat sessions, `data/attachments/{slug}/`, `data/quotes/{slug}/`.

**Kept:** tenant row, connectors, integrations, `ingested_files`, LanceDB vectors, document/catalog files, `mail_imap_uids` (old inbox messages are not re-fetched).

---

## Sync documents (RAG)

```bash
chatbot sync TENANT_SLUG path/to/docs
chatbot sync TENANT_SLUG --fresh
# Docker:
./sail chatbot sync TENANT_SLUG /app/data/docs
```

Supported: `.md`, `.docx`, `.pdf`, `.csv`, `.xlsx`, `.xls`. Enable RAG per tenant in dashboard or `RAG_ENABLED` default in `.env`.

Uploaded files live under `data/docs/{slug}/`. The dashboard **Sync** button and `chatbot sync` only touch that folder — not the ERPNext catalog (see below).

**Creole rewrite gate:** `RAG_REWRITE_LANG_FILTER=true` limits query rewrite to messages matching Creole markers (`creole_script_heuristic.py`). Set `RAG_VERBOSE=true` for logs.

---

## ERPNext catalog → RAG

When an **ERPNext integration** is active, the dashboard lets you export the live item catalogue (price, optional stock) into the bot knowledge base:

1. **Integrations → ERPNext** — enable **Sync catalog to RAG**, set **Catalog sync interval (minutes)** (default 360), **Catalog price list** (default `Standard Selling`), optionally disable **Include stock totals**.
2. Save the integration. The `worker-catalog` service polls every `CATALOG_POLL_SECONDS` (default 300) and runs a sync when the interval has elapsed.
3. Or click **Sync catalog now** (runs in the background; refresh the page for last sync / item count / error).

Each active item becomes one markdown file under `data/catalog/{slug}/{item_code}.md`. The RAG pipeline re-embeds **only files whose content changed** (content hash). Stock and price in RAG are a **snapshot** at sync time; live quotes still use hooks + `ProductResolver`.

**Resume a partial index** (after 429/503 or interrupted rebuild) — same pipeline as dashboard/worker, with progress bar:

```bash
./sail chatbot catalog-rag rebuild my-bot              # missing/changed files only
./sail chatbot catalog-rag rebuild my-bot --dry-run    # show counts, no API calls
./sail chatbot catalog-rag rebuild my-bot --all        # scan all catalog .md (unchanged skip embed)
./sail chatbot catalog-rag sync my-bot                 # ERPNext fetch + RAG (dashboard equivalent)
```

**Embedding rate limits:** all RAG paths share `GeminiEmbedder` — max speed until Google returns 429/503, then adaptive backoff (`Retry-After` header or exponential). Optional env: `EMBED_RETRY_MAX` (default 5), `EMBED_RETRY_BASE_429_SECONDS` (30), `EMBED_RETRY_BASE_503_SECONDS` (5).

**Pricing in catalog files:** paginated fetch from ERPNext **Item Price** (configured price list), fallback to `Item.standard_rate`, otherwise `Price: not available` — never `0.0`.

Optional env: `CATALOG_POLL_SECONDS` — catalog worker poll interval (not the per-bot sync interval, which is in integration config).

**Monitoring env (optional):**

| Variable | Purpose |
|----------|---------|
| `GEMINI_PRICING_JSON` | JSON override of per-model $/1M token rates (merged over built-in defaults) |
| `CLIENT_BILLING_INPUT_PER_MILLION_USD` | Flat client billable input rate (default 1.0) |
| `CLIENT_BILLING_OUTPUT_PER_MILLION_USD` | Flat client billable output rate (default 3.0) |
| `DISK_SNAPSHOT_ENABLED` | Nightly disk snapshots via catalog worker (default true) |

```bash
# One-off sync for all due tenants (inside container):
python -m chatbot.interfaces.worker_catalog --once
```

---

## Monitoring & usage billing

Gemini API calls (chat, rewrite, embed) are metered into `api_usage_daily`. Disk usage is scanned on demand and snapshotted nightly into `disk_usage_daily` by `worker-catalog`.

| View | Who | What |
|------|-----|------|
| `/dashboard/monitoring` | `admin` | All bots — token chart, disk trends, internal + client cost per bot |
| `?tab=monitoring` on bot detail | `admin`, `client_admin` | Per-bot charts, usage table, live disk; admins can set client billing $/M |

**Cost tiers:** admins see **internal** estimates from Google list prices (per model; default chat model `gemini-2.5-flash`). **Client billable** uses a single flat input/output $/M rate — platform default in `.env` or per-bot override on the admin monitoring form (`tenants.client_billing_*`, not in bot config JSON).

Disk history charts need at least one worker snapshot; restart `worker-catalog` after deploy if charts are empty.

---

## Tests

```bash
pytest
# Docker:
./sail test
```

---

## Production (Ubuntu + systemd)

Bind uvicorn to `127.0.0.1:8000`, put **nginx** or **Caddy** in front for TLS. Run **worker-automation**, **worker-mail**, and **worker-catalog** (when ERPNext catalog sync is used) as separate systemd units (same image/env as API). Use **MySQL** (`DATABASE_URL=mysql+pymysql://…`). See [ARCHITECTURE.md](ARCHITECTURE.md) for hooks and multi-tenant layout.

Public webhook examples:

- `https://your-host/webhooks/whatsapp/{slug}`
- `https://your-host/webhooks/messenger/{slug}`
- `https://your-host/webhooks/instagram/{slug}`

Subscribe Meta **`messages`** field on each channel. Configure verify token and app secret in **Connectors** for that `{slug}`.

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — layers, hooks, workers, settings
- [AGENTS.md](AGENTS.md) — conventions for coding agents
- [docs/dev/greenmail.md](docs/dev/greenmail.md) — local email testing (GreenMail + Mailpit)

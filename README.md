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
| http://localhost:80/… | Same via Caddy (optional) |
| `POST /c/{slug}/chat` | Chat API (`Authorization: Bearer <tenant_token>`) |

**Sail commands**

```bash
./sail up -d          # start db, api, worker-automation, caddy
./sail down           # stop stack
./sail logs           # follow logs
./sail shell          # bash in API container
./sail mysql          # MySQL CLI
./sail chatbot …      # CLI (tenant-create, sync, user-create, …)
./sail migrate        # alembic upgrade head
./sail test           # pytest in container
./sail worker-logs    # automation worker logs
```

Run any Compose command: `./sail ps`, `./sail restart api`, etc.

**No rebuild for code edits:** `docker-compose.override.yml` mounts `./src` into the API (and worker); uvicorn runs with `--reload`. Rebuild only when `pyproject.toml` or the Dockerfile change: `./sail build api`.

**First-time secrets in `.env`**

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM + embeddings (fallback if tenant has no override) |
| `ADMIN_TOKEN` | `/admin/*` API + Streamlit legacy admin pages |
| `APP_SECRET_KEY` | Fernet — encrypts connector configs in DB |
| `SESSION_SECRET` | Dashboard session cookies |

Generate Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Automation worker** processes `hook_events` (orders, etc.). Compose starts `worker-automation` automatically; locally: `python -m chatbot.interfaces.worker_automation`.

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

See [docs/dev/greenmail.md](docs/dev/greenmail.md) for connector presets (IN: GreenMail, OUT: Mailpit).

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
- Worker (separate terminal): `python -m chatbot.interfaces.worker_automation`
- Migrations (MySQL only): `alembic upgrade head` — tests use SQLite `create_all`, no Alembic

**`.env` hot reload:** most vars reload when `.env` is saved (mtime). **`DATABASE_URL`** still needs an API restart.

---

## Test chat

Use the dashboard **Test chat** tab on a bot detail page (server-side `ChatService`). For API integration tests, call `POST /c/{slug}/chat` with the tenant Bearer token.

---

## Sync documents (RAG)

```bash
chatbot sync TENANT_SLUG path/to/docs
chatbot sync TENANT_SLUG --fresh
# Docker:
./sail chatbot sync TENANT_SLUG /app/data/docs
```

Supported: `.md`, `.docx`, `.pdf`, `.csv`, `.xlsx`, `.xls`. Enable RAG per tenant in dashboard or `RAG_ENABLED` default in `.env`.

**Creole rewrite gate:** `RAG_REWRITE_LANG_FILTER=true` limits query rewrite to messages matching Creole markers (`creole_script_heuristic.py`). Set `RAG_VERBOSE=true` for logs.

---

## Tests

```bash
pytest
# Docker:
./sail test
```

---

## Production (Ubuntu + systemd)

Bind uvicorn to `127.0.0.1:8000`, put **nginx** or **Caddy** in front for TLS. Run the **automation worker** as a second systemd unit (same image/env as API). Use **MySQL** (`DATABASE_URL=mysql+pymysql://…`). See [ARCHITECTURE.md](ARCHITECTURE.md) for hooks and multi-tenant layout.

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

# Gemini customer chatbot

Python 3.12, FastAPI + Gemini, optional RAG (LanceDB), WhatsApp Cloud API webhook. **Messenger and Instagram** are not wired in code yet; the sections below describe **Meta-only** setup if you add matching webhook routes later (same `ChatService` pattern as WhatsApp).

## Environment (pyenv)

```bash
pyenv install 3.12.8   # or another 3.12.x you prefer
pyenv virtualenv 3.12.8 chatbot
pyenv local chatbot
pip install -e ".[dev]"
cp .env.example .env
# set GEMINI_API_KEY and other vars in .env
```

The repo includes [`.python-version`](.python-version) with the virtualenv name `chatbot` (pyenv selects it when the env exists).

## Run API

```bash
mkdir -p data
uvicorn chatbot.interfaces.api.main:app --reload
```

**`.env` without restarting:** `CHAT_MODEL`, `REWRITE_MODEL`, `EMBEDDING_MODEL`, `GEMINI_API_KEY`, RAG flags, etc. are re-read when the `.env` file is **saved** (mtime change). Gemini clients and the RAG rewrite gate refresh on the next HTTP request. You still need a **full restart** if you change **`DATABASE_URL`** (the SQLAlchemy engine is created once at startup).

`uvicorn --reload` only watches Python files by default; that is separate from `.env` hot reload above.

## Test UI (Streamlit)

```bash
streamlit run apps/streamlit_chat.py
```

Point the sidebar API URL at `http://127.0.0.1:8000` if needed. For **`CHAT_API_SECRET`**, the app loads the **repository root `.env`** at startup (then `os.environ`, then **`st.secrets`**) so the same `.env` as the API usually works locally.

## Channel text rendering

Meta channels (Messenger/Instagram) and WhatsApp do not render markdown the same way. To reduce client-to-client differences:

- **Messenger/Instagram** outbound text is normalized to plain readable text (strip markdown markers, normalize bullets/spacing).
- **WhatsApp** keeps simple emphasis-compatible text (e.g. `*bold*`, `_italic_`) while cleaning malformed marker patterns.

You may still see minor visual differences between Messenger web and mobile clients, but raw markdown markers (for example `* *Title`) should no longer leak into outbound messages.

## Sync documents (RAG)

Reconcile a folder with the index: remove vectors and DB rows for files that were ingested under that root but no longer exist on disk, then (re)ingest every supported file there (unchanged files are skipped by hash). Supported types include **markdown** (`.md`), Word (`.docx`), PDF, CSV, and Excel (`.xlsx`, `.xls`).

```bash
python -m chatbot sync path/to/file_or_dir
```

To **replace the entire RAG index** (e.g. when switching from one client doc folder to another), clear all vectors and ingest metadata first, then ingest only the given path:

```bash
python -m chatbot sync --fresh docs/vdtech
```

This does not delete chat messages or orders in `app.db`—only LanceDB chunks and `ingested_files` rows. Restart the API after a sync if it was already running.

On PEP 668–managed Python (e.g. Homebrew), use a venv (`python3 -m venv .venv && source .venv/bin/activate`) or pyenv `chatbot` before `pip install`.

Enable RAG in `.env`: `RAG_ENABLED=true`.

**Sources in replies:** With `DEV_MODE=false` (typical production), retrieved chunks are passed to the model **without** file paths, and the assistant is told not to add `(Source: …)` lines. Set `DEV_MODE=true` for local debugging if you want paths in context and traceable filenames in answers.

### Creole marker gate for query rewrite

When `RAG_REWRITE_LANG_FILTER=true`, the optional **LLM rewrite** step (see `RAG_REWRITE_ENABLED`) is allowed **only** if a **Creole marker token** matches in the user text. Markers and tokenization live in [`creole_script_heuristic.py`](src/chatbot/adapters/rag/creole_script_heuristic.py) (`CREOLE_MARKERS`, whole-word match). There is **no fastText** dependency for this gate.

- **`RAG_REWRITE_LANG_FILTER=false`**: no language gate — rewrite follows `RAG_REWRITE_ENABLED` only (same as before).
- **`RAG_REWRITE_LANG_FILTER=true`**: rewrite runs only when `creole_markers_hit(user_text)` is true.

Short markers (e.g. `la`, `sa`) can appear in French; extend or trim `CREOLE_MARKERS` for your traffic. To try sample lines locally: `pytest tests/test_lid_creole_sentence_probe.py -v -s`.

Set **`RAG_VERBOSE=true`** to log marker hits and rewrite decisions. Restart the API after changing gate-related code (or rely on `.env` mtime reload for flags only).

## Tests

```bash
pytest
```

## Production setup (Ubuntu)

Example layout: app code under `/home/YOUR_LINUX_USER/chatbot`, systemd runs **uvicorn on `127.0.0.1:8000` only**, **nginx** terminates TLS and proxies to that socket. Replace `YOUR_LINUX_USER` and `chatbot.example.com` with your values.

### Python 3.12 next to system Python (e.g. Ubuntu 22.04)

The project needs **Python ≥ 3.12** ([`pyproject.toml`](pyproject.toml)). Installing 3.12 does not replace the system `python3` (often 3.10); use a **dedicated venv** for this app:

```bash
sudo apt update
sudo apt install -y software-properties-common git build-essential
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

### Clone, venv, install

```bash
sudo mkdir -p /home/YOUR_LINUX_USER/chatbot
sudo chown YOUR_LINUX_USER:YOUR_LINUX_USER /home/YOUR_LINUX_USER/chatbot
# as YOUR_LINUX_USER:
cd /home/YOUR_LINUX_USER/chatbot
git clone <your-repo-url> .
python3.12 -m venv .venv --prompt chatbot
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
mkdir -p data
cp .env.example .env
# edit .env (GEMINI_API_KEY, DATABASE_URL, WHATSAPP_*, RAG flags, etc.)
```

### nginx and Certbot (HTTPS)

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/chatbot` (HTTP first; Certbot will add TLS):

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name chatbot.example.com;

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

Enable, test, obtain certificates:

```bash
sudo ln -sf /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d chatbot.example.com
```

After Certbot, confirm the **443** server still `proxy_pass`es to `http://127.0.0.1:8000`. Optional: add `client_max_body_size 20m;` inside `server { }` if you expect large webhooks or uploads.

**Firewall:** allow **80** and **443**; do **not** expose port **8000** publicly if nginx is the front door (`sudo ufw allow 'Nginx Full'` or equivalent).

### systemd (start on boot)

`/etc/systemd/system/chatbot.service`:

```ini
[Unit]
Description=Chatbot FastAPI (uvicorn)
After=network.target

[Service]
Type=simple
User=YOUR_LINUX_USER
Group=YOUR_LINUX_USER
WorkingDirectory=/home/YOUR_LINUX_USER/chatbot
Environment=PATH=/home/YOUR_LINUX_USER/chatbot/.venv/bin
ExecStart=/home/YOUR_LINUX_USER/chatbot/.venv/bin/uvicorn chatbot.interfaces.api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`WorkingDirectory` must be the repo root so `.env` and relative paths like `./data/` resolve the same way as locally (see [`.env.example`](.env.example)).

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now chatbot.service
sudo systemctl status chatbot.service
journalctl -u chatbot.service -f
```

### Reload and stop (operations)

| Goal | Command |
|------|---------|
| **Apply code or dependency changes** (after `git pull`, `pip install`, edits under `src/`) | `sudo systemctl restart chatbot.service` |
| **After editing the `.service` unit file** | `sudo systemctl daemon-reload` then `sudo systemctl restart chatbot.service` |
| **Many `.env` changes** (models, keys, RAG flags) | Save `.env`; the app reloads clients on the **next request** when the file mtime changes. **Exception:** changing **`DATABASE_URL`** still needs **`sudo systemctl restart chatbot.service`**. |
| **Stop the API** | `sudo systemctl stop chatbot.service` |
| **Start the API** | `sudo systemctl start chatbot.service` |
| **nginx config change** | `sudo nginx -t && sudo systemctl reload nginx` |

Meta callback URLs use your public origin, e.g. `https://chatbot.example.com/webhooks/whatsapp`, `https://chatbot.example.com/webhooks/messenger`, and `https://chatbot.example.com/webhooks/instagram`.

### Security (what this repo does and does not do)

- **`POST /v1/chat`:** set **`CHAT_API_SECRET`** in `.env` to a long random value. When non-empty, the API requires **`Authorization: Bearer <same value>`**. Leave it empty for local dev without a header. The Streamlit test app reads **`CHAT_API_SECRET`** from the **repo root `.env`** (loaded on startup), then the process environment, then **`.streamlit/secrets.toml`**. Meta webhooks do **not** use this header; they use **`/webhooks/...`** with Meta’s own verification.
- **`GET /healthz`** is intentionally open for probes.
- **Meta webhooks:** `GET` verification compares `hub.verify_token` with channel tokens (`WHATSAPP_VERIFY_TOKEN`, `MESSENGER_VERIFY_TOKEN`, `INSTAGRAM_VERIFY_TOKEN`; Messenger/Instagram fall back to `WHATSAPP_VERIFY_TOKEN` when blank). **`POST`** payloads are checked against **`X-Hub-Signature-256`** using **`WHATSAPP_APP_SECRET`** (Meta App Secret). These channels do not call `/v1/chat`.
- **Hardening baseline:** uvicorn bound to **127.0.0.1**; TLS on **nginx**; firewall; keep secrets only in `.env` with tight file permissions (`chmod 600 .env`). Optional: nginx **`limit_req`** for extra abuse protection.

Securing `/v1/chat` does **not** add a login to the Streamlit **browser UI**; restrict who can open Streamlit separately (private app, SSO, or nginx) if needed.

## WhatsApp (dev)

**Meta ([developers.facebook.com](https://developers.facebook.com/))**

1. **App** → add product **WhatsApp** (Cloud API).
2. **WhatsApp → API Setup:** copy **Temporary access token** → `WHATSAPP_ACCESS_TOKEN`, **Phone number ID** → `WHATSAPP_PHONE_NUMBER_ID`. Under test tools, **add and verify** every phone number that may chat with the sandbox (SMS/WhatsApp code).
3. **App → Settings → Basic:** copy **App secret** → `WHATSAPP_APP_SECRET` (needed so `POST` webhooks pass `X-Hub-Signature-256` checks).
4. **WhatsApp → Configuration:** set **Callback URL** to `https://<your-public-host>/webhooks/whatsapp` (path required; not `/` alone). Set **Verify token** to the same random string as `WHATSAPP_VERIFY_TOKEN` in `.env`, then **Verify and save**.
5. Still on **Configuration**, under **Webhook fields**, **activate** (subscribe to) **`messages`** — use the toggle or **Manage** so the field shows **Subscribed**. This is easy to skip: until `messages` is on, Meta only hits your URL for webhook **verification** (`GET`); inbound chats use **`POST`** and will not appear at all.

**This machine**

- Run the API (`uvicorn` defaults to port **8000**). Run **`ngrok http 8000`** (or your port). Put the ngrok **https** forwarding URL + `/webhooks/whatsapp` in step 4; free ngrok URLs change when ngrok restarts—update Meta each time.
- Optional: open `http://127.0.0.1:4040` to confirm **`POST /webhooks/whatsapp`** when you send a WhatsApp message **to the test business number** shown in API Setup (your phone is the customer; that number is the “shop” line).

`.env` keys: see [`.env.example`](.env.example) (`WHATSAPP_*`).

## Facebook Messenger — Page (Meta setup)

Use the **same** Meta app or a dedicated one; this is the **Messenger** product (Facebook **Page** inbox), not WhatsApp.

1. **App** → add product **Messenger**.
2. **Messenger → Settings / API Setup:** connect the **Facebook Page** the bot should use. Create or copy a **Page access token** (long-lived in production). You send replies with **`POST https://graph.facebook.com/v…/me/messages`** and that token — different from WhatsApp’s `phone_number_id` + WhatsApp token.
3. **Messenger → Webhooks** (or **App → Webhooks** with Messenger fields): set **Callback URL** to `https://<host>/webhooks/messenger`, **Verify token** to `MESSENGER_VERIFY_TOKEN` (or `WHATSAPP_VERIFY_TOKEN` if you leave `MESSENGER_VERIFY_TOKEN` blank), same `hub.challenge` flow as WhatsApp on `GET`.
4. **Subscribe** the **Page** to webhook fields: at minimum **`messages`** (and **`messaging_postbacks`** if you use buttons). **Activate** them like WhatsApp’s `messages` field — otherwise you only see verification `GET`s, not chat `POST`s.
5. **App Review:** grant **`pages_messaging`** (and related Page permissions) for users who are not admins/testers of the app.

Inbound `POST` bodies use **`messaging[]`** / `sender.id` (PSID), not WhatsApp’s JSON shape. Signature: **`X-Hub-Signature-256`** with the same **App secret** as in App Settings (`WHATSAPP_APP_SECRET` here). Replies are sent with **`MESSENGER_PAGE_ACCESS_TOKEN`** to `https://graph.facebook.com/v21.0/me/messages`.

## Instagram — DMs (Meta setup)

Instagram customer chat uses the **Instagram API with Instagram Login** (or legacy Basic Display flows are being retired — follow Meta’s current “Instagram messaging” docs). Typically:

1. The Instagram account is a **Professional** (Business/Creator) account **linked** to a **Facebook Page** you control.
2. **App** → add **Instagram** product; complete **Instagram Login** / messaging configuration in the dashboard (scopes such as `instagram_manage_messages` where applicable — names change; check Meta’s checklist).
3. **Webhooks:** register callback URL `https://<host>/webhooks/instagram` for **Instagram** object fields (e.g. **`messages`** for DMs). Verify token is `INSTAGRAM_VERIFY_TOKEN` (or `WHATSAPP_VERIFY_TOKEN` if blank), and you must subscribe the needed fields.
4. Replies use `https://graph.instagram.com/v25.0/{INSTAGRAM_IG_USER_ID}/messages` with **`INSTAGRAM_ACCESS_TOKEN`** and the inbound `sender.id` (IGSID), not the WhatsApp send URL.

Policies (24-hour session, human agent handoff, etc.) differ from WhatsApp; read Meta’s **Instagram messaging** policy pages before production.

---

Webhook routes implemented in this repo: `/webhooks/whatsapp`, `/webhooks/messenger`, `/webhooks/instagram`.

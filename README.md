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

Point the sidebar API URL at `http://127.0.0.1:8000` if needed.

## Sync documents (RAG)

Reconcile a folder with the index: remove vectors and DB rows for files that were ingested under that root but no longer exist on disk, then (re)ingest every supported file there (unchanged files are skipped by hash). Supported types include **markdown** (`.md`), Word (`.docx`), PDF, CSV, and Excel (`.xlsx`, `.xls`).

```bash
python -m chatbot sync path/to/file_or_dir
```

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
3. **Messenger → Webhooks** (or **App → Webhooks** with Messenger fields): set **Callback URL** (e.g. `https://<host>/webhooks/messenger` — must match what you implement), **Verify token** (your string; same `hub.challenge` flow as WhatsApp on `GET`).
4. **Subscribe** the **Page** to webhook fields: at minimum **`messages`** (and **`messaging_postbacks`** if you use buttons). **Activate** them like WhatsApp’s `messages` field — otherwise you only see verification `GET`s, not chat `POST`s.
5. **App Review:** grant **`pages_messaging`** (and related Page permissions) for users who are not admins/testers of the app.

Inbound `POST` bodies use **`messaging[]`** / `sender.id` (PSID), not WhatsApp’s JSON shape. Signature: **`X-Hub-Signature-256`** with the same **App secret** as in App Settings.

## Instagram — DMs (Meta setup)

Instagram customer chat uses the **Instagram API with Instagram Login** (or legacy Basic Display flows are being retired — follow Meta’s current “Instagram messaging” docs). Typically:

1. The Instagram account is a **Professional** (Business/Creator) account **linked** to a **Facebook Page** you control.
2. **App** → add **Instagram** product; complete **Instagram Login** / messaging configuration in the dashboard (scopes such as `instagram_manage_messages` where applicable — names change; check Meta’s checklist).
3. **Webhooks:** register a **Callback URL** for **Instagram** object fields (e.g. **`messages`** for DMs). Same verify-token + challenge pattern; **subscribe** the fields you need or inbound events will not `POST`.
4. Replies use the **Instagram messaging** Graph endpoints with the correct **page/IG user** identifiers (IGSID for conversations), not the WhatsApp send URL.

Policies (24-hour session, human agent handoff, etc.) differ from WhatsApp; read Meta’s **Instagram messaging** policy pages before production.

---

For **Messenger** and **Instagram**, implement separate FastAPI routes and adapters (parse their payloads, call `handle_user_message` with a distinct `session_id` prefix, send via the right Graph API). This repo currently includes **WhatsApp** only (`/webhooks/whatsapp`).

# Gemini customer chatbot

Python 3.12, FastAPI + Gemini, optional RAG (LanceDB), WhatsApp Cloud API webhook.

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

Reconcile a folder with the index: remove vectors and DB rows for files that were ingested under that root but no longer exist on disk, then (re)ingest every supported file there (unchanged files are skipped by hash). Supported types include **markdown** (`.md`), PDF, CSV, and Excel (`.xlsx`, `.xls`).

```bash
python -m chatbot sync path/to/file_or_dir
```

On PEP 668–managed Python (e.g. Homebrew), use a venv (`python3 -m venv .venv && source .venv/bin/activate`) or pyenv `chatbot` before `pip install`.

Enable RAG in `.env`: `RAG_ENABLED=true`.

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

Set `WHATSAPP_*` variables, expose `https://.../webhooks/whatsapp` (e.g. ngrok), configure Meta webhook verify token to match `WHATSAPP_VERIFY_TOKEN`.

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

**`.env` without restarting:** `CHAT_MODEL`, `REWRITE_MODEL`, `EMBEDDING_MODEL`, `GEMINI_API_KEY`, RAG / FastText flags, etc. are re-read when the `.env` file is **saved** (mtime change). Gemini clients and the FastText gate refresh on the next HTTP request. You still need a **full restart** if you change **`DATABASE_URL`** (the SQLAlchemy engine is created once at startup).

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

### FastText gating for query rewrite (English / Creole only)

When `RAG_REWRITE_LANG_FILTER=true`, the optional **LLM rewrite** step (see `RAG_REWRITE_ENABLED`) runs only if [fastText language id](https://fasttext.cc/docs/en/language-identification.html) says the user text is:

- **English** (`en`) with score ≥ `RAG_REWRITE_MIN_PROB_EN`, or  
- a **Creole** code listed in `RAG_REWRITE_CREOLE_LABELS` (comma-separated ISO 639-1 labels, default `ht` for Haitian Creole in `lid.176`), or  
- **French** with score strictly below `RAG_REWRITE_FR_MAX_PROB_CREOLE` (low-confidence `fr` is treated as Creole-like and rewrite is allowed).

Otherwise the **raw user message** is used for embedding search (no rewrite call).

1. Download a lid model, e.g. `lid.176.bin` from [fastText supervised models](https://fasttext.cc/docs/en/supervised-models.html#content) (identification section).
2. Set `FASTTEXT_LID_MODEL_PATH=./models/lid.176.bin` (or an absolute path).
3. Set `RAG_REWRITE_LANG_FILTER=true` and restart the API.

Set **`RAG_VERBOSE=true`** to print RAG decisions to the console (FastText top‑3 + rule outcome, whether the LLM rewrite runs, embed/search hit summaries). Restart the API after changing it.

## Tests

```bash
pytest
```

## WhatsApp (dev)

Set `WHATSAPP_*` variables, expose `https://.../webhooks/whatsapp` (e.g. ngrok), configure Meta webhook verify token to match `WHATSAPP_VERIFY_TOKEN`.

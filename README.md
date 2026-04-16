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

## Test UI (Streamlit)

```bash
streamlit run apps/streamlit_chat.py
```

Point the sidebar API URL at `http://127.0.0.1:8000` if needed.

## Ingest documents (RAG)

```bash
python -m chatbot ingest path/to/file_or_dir
```

On PEP 668–managed Python (e.g. Homebrew), use a venv (`python3 -m venv .venv && source .venv/bin/activate`) or pyenv `chatbot` before `pip install`.

Enable RAG in `.env`: `RAG_ENABLED=true`.

## Tests

```bash
pytest
```

## WhatsApp (dev)

Set `WHATSAPP_*` variables, expose `https://.../webhooks/whatsapp` (e.g. ngrok), configure Meta webhook verify token to match `WHATSAPP_VERIFY_TOKEN`.

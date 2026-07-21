# CAC40 mean-reversion bot (Evenor)

Hedge-mode CAC40 bot: IG connectors, multimodal Gemini decisions, HedgeLedger backtest, Fund Manager heartbeat (`source=evenor`).

## Layout

| Path | Role |
|------|------|
| `src/chatbot/cac40/` | Core engine (IG, charts, LLM, RiskGate, HedgeLedger, backtest, live scheduler) |
| `prompts/cac40/system.md` | LLM system prompt |
| `/dashboard/bots/{slug}/cac40` | Backtest UI integration |
| `/dashboard/bots/{slug}/cac40/runs/{id}` | Run report + LLM decision browser (charts sent to Gemini) |
| Connectors type `ig` | IG credentials (bidirectional) |
| Integration type `cac40_backtest` | FM URL/token + defaults |
| `worker-cac40-ohlc` | Background OHLC top-up from IG |

## CLI

```bash
# Backtest (replay = no Gemini calls unless cache filled)
PYTHONPATH=src python -m chatbot cac40 backtest path/to/cac40_15m.csv --llm-mode replay

# Live / demo single cycle (dry-run default)
PYTHONPATH=src python -m chatbot cac40 live --once --dry-run

# OHLC sync worker (once)
PYTHONPATH=src python -m chatbot.interfaces.worker_cac40_ohlc --once
```

## Fund Manager

Heartbeat: `POST /jessebot/notify-up` with `source=evenor` and optional `X-Notify-Token`.
UI: **Jesse & Evenor** menu — expandable multi-leg PnL for Evenor rows.

## Data

### Incremental updates (IG)

On **CAC40 Backtest** click **Sync from IG** (requires an active IG connector):
- Appends 15m mid-price bars since the last stored timestamp
- Gap &gt; 60 days is rejected — re-upload a futures CSV first
- Empty dataset: manual sync bootstraps ~60 days from IG
- Background worker (`worker-cac40-ohlc`, poll `CAC40_OHLC_POLL_SECONDS` default 900) tops up tenants with active CAC40 + IG + existing CSV (no cron bootstrap of empty files)

The OHLC card shows **last candle**, **age in hours**, last sync (manual/worker), fetch window, and worker cycle time.

### How to verify it works

1. Active IG connector + active CAC40 integration + OHLC CSV present (upload BacktestMarket if needed).
2. Manual: open `/dashboard/bots/{slug}/cac40` → **Sync from IG**. Expect green flash, `trigger=manual`, **Last candle** updated (or `+0 bars` if already fresh).
3. Worker once:
   ```bash
   ./sail up -d worker-cac40-ohlc
   ./sail exec worker-cac40-ohlc python -m chatbot.interfaces.worker_cac40_ohlc --once
   ./sail logs worker-cac40-ohlc
   ```
   Refresh the page: **Worker / cron → Last cycle** should show a recent `finished_at`, and bot sync status `trigger=worker`.
4. Files on disk:
   - `data/cac40/{slug}/ohlc/cac40_15m.csv` — last rows = last candles
   - `data/cac40/{slug}/ohlc/sync_status.json` — per-bot sync result
   - `data/cac40/worker_status.json` — last worker cycle

### Longer history

For multi-year backtests, buy a futures CSV (e.g. [backtestmarket CAC MX 15m](https://www.backtestmarket.com/en/cac-40-15m)) and use **Upload CSV** with source **BacktestMarket (MX 15m)** — Evenor converts `;` / no-header files automatically.

Each bot has its own OHLC file under `data/cac40/{slug}/ohlc/` (`ohlc_15m.csv`, or legacy `cac40_15m.csv`). Set **symbol** + **IG epic** on the CAC40 integration (Integrations tab) — one bot = one symbol; create another bot for a different market. Match the IG connector epic to that bot's symbol for Sync from IG.

You can also place a CSV at that path (columns `Date,Open,High,Low,Close[,Volume]`).

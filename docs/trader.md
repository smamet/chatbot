# Trader bots (Evenor)

Hedge-mode trading bots: IG connectors, multimodal Gemini decisions, HedgeLedger backtest, Fund Manager heartbeat (`source=evenor`).

Bots are typed: `tenants.bot_type = trader` (not an Integrations row). One bot = one market (symbol/epic). Use **market profiles** (`cac40`, `eurusd`, …) for defaults + calendar + prompt.

## Layout

| Path | Role |
|------|------|
| `src/chatbot/trader/` | Core engine (IG, charts, LLM, RiskGate, HedgeLedger, backtest, live scheduler) |
| `prompts/trader/profiles/{id}.md` | Default LLM system prompts per market profile |
| `/dashboard/bots/{slug}?tab=trading` | Trading UI (Data / Live / Backtest) — default for trader bots |
| Connectors type `ig` | IG credentials (multiple accounts; first selected = price feed) |
| `worker-trader-ohlc` | Background OHLC top-up from IG REST `/prices` (off bots / gaps) |
| `worker-trader-live` | Background live cycles |
| `worker-trader-stream` | Lightstreamer: live ticks → CSV bars + TRADE wake-up book sync |
| `data/trader/{slug}/` | OHLC, backtests, live journal (auto-migrates from `data/cac40/{slug}/`) |

## Bot type + config

1. Create bot with type **Trader**, or migrate via Alembic `026` (active `cac40_backtest` → `bot_type=trader` + `config.trader`).
2. **Config → Trading settings**: market profile, symbol, epic, Fund Manager, max legs.
3. **Config → Trading system prompt**: used by Gemini. Live/Backtest do **not** override it. **Load default prompt** fills from the profile file; save to apply.
4. Empty prompt → profile default file.

## Market profiles

| Id | Symbol default | Calendar | Prompt |
|----|----------------|----------|--------|
| `cac40` | CAC40 / `IX.D.CAC.BMU.IP` | Euronext FR40 (IG France 40 Cash CFD hours) | `prompts/trader/profiles/cac40.md` |
| `eurusd` | EURUSD / `CS.D.EURUSD.MINI.IP` | IG FX 24x5 approx | `prompts/trader/profiles/eurusd.md` |

Confirm epics on your IG account (Demo vs Live differ).

## Live trading

On **Trading → Live**:

1. Add IG connector(s); check accounts to use (first = primary price feed).
2. Save strategy; **Off** skips the bot; **Live** places working orders on selected accounts (use DEMO for paper).
3. Arm waits for the next 15m Paris slot; **Run cycle now** + **Force LLM** bypasses the LLM schedule.
4. Disarming stops new cycles but does **not** cancel resting IG orders.

Weekend/holiday flatten uses the bot’s **calendar_id** from its market profile (FR40 Euronext holidays vs FX weekend-only).

## Workers / CLI

```bash
./sail up -d worker-trader-ohlc worker-trader-live worker-trader-stream
./sail exec worker-trader-live python -m chatbot.interfaces.worker_trader_live --once
./sail logs worker-trader-stream

# CLI
PYTHONPATH=src python -m chatbot trader backtest path/to.csv --llm-mode replay
PYTHONPATH=src python -m chatbot trader live --once --dry-run
PYTHONPATH=src python -m chatbot trader migrate-data   # data/cac40 → data/trader

# DEMO Lightstreamer probe / order matrix
PYTHONPATH=src python -m chatbot trader stream-probe --from-db-slug my-trader --seconds 45
PYTHONPATH=src python -m chatbot trader order-probe --from-db-slug my-trader
```

### Lightstreamer live path (armed bots)

While mode is **Live**, `worker-trader-stream` keeps a Lightstreamer session on the **primary** IG connector:

- **PRICE ticks** → `stream_quote.json` + closed synthetic 15m bars appended to the local CSV (no hot-path `/prices`)
- **TRADE** events → debounced REST `GET /positions` + `/workingorders` → existing `replace_open` book sync under the same `.cycle.lock` as live cycles
- **Health:** `data/trader/{slug}/stream_status.json` + global `stream_worker_status.json`; Trading tab shows stream ok / stale / down. Stale stream fail-closes LLM.
- **Gap repair (auto):** when ticks recover or a bar gap is detected, the stream worker rate-limits a REST OHLC top-up (`STREAM_GAP_REPAIR_RETRY_SECONDS`, default 60). If **Data → Sync from IG** already caught the CSV up, the next healthy heartbeat clears `gap_repair_*` without another `/prices` call (no worker restart needed).
- **Backfill / off bots:** still REST `/prices` via Sync or `worker-trader-ohlc`
- Shared CST/XST cache avoids dual `/session` logins from stream + live workers

Pin `lightstreamer-client-lib==1.0.3`. Orders are never placed over Lightstreamer.

Poll: `TRADER_OHLC_POLL_SECONDS` (900), `TRADER_LIVE_POLL_SECONDS` (60), `TRADER_STREAM_LOOP_SECONDS` (5), `STREAM_TICK_STALE_SECONDS` (120). Legacy `CAC40_*` env names still accepted.

## Legacy URLs

`/dashboard/bots/{slug}/cac40…` redirects to `/trader…` or `?tab=trading`.

## Fund Manager

Heartbeat: `POST …/jessebot/notify-up` with `source=evenor` and optional `X-Notify-Token`.

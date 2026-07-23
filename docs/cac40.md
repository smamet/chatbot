# CAC40 mean-reversion bot (Evenor)

Hedge-mode CAC40 bot: IG connectors, multimodal Gemini decisions, HedgeLedger backtest, Fund Manager heartbeat (`source=evenor`).

## Layout

| Path | Role |
|------|------|
| `src/chatbot/cac40/` | Core engine (IG, charts, LLM, RiskGate, HedgeLedger, backtest, live scheduler) |
| `prompts/cac40/system.md` | LLM system prompt |
| `/dashboard/bots/{slug}/cac40` | Backtest UI + **Live trading** panel |
| `/dashboard/bots/{slug}/cac40/live/report` | Paper/live results (decision browser + PnL) |
| `/dashboard/bots/{slug}/cac40/runs/{id}` | Backtest run report + LLM decision browser |
| Connectors type `ig` | IG credentials (multiple accounts allowed; optional label) |
| Integration type `cac40_backtest` | FM URL/token + defaults |
| `worker-cac40-ohlc` | Background OHLC top-up from IG |
| `worker-cac40-live` | Background live/paper cycles |

## Live trading (dashboard)

On **CAC40 Backtest** → **Live trading**:

1. Add one or more IG connectors (Connectors tab). Use a label (e.g. Demo CFD).
2. Check the accounts to use; **first checked = primary price feed**. Orders in Live mode are mirrored to every selected account.
3. Save strategy (same knobs as the simulation form, minus period/llm_mode).
4. **Paper** = Gemini + simulated ledger (no IG order API). **Live** = real working orders (each connector’s DEMO/LIVE env). **Off** = worker skips the bot.
5. **View results** opens the live report (same decision browser as backtest: charts sent to Gemini + LLM answer). Recent cycles also appear under **Live cycles** on the CAC40 page.
6. Disarming (Off/Paper) stops new cycles but **does not cancel** IG working orders already placed.

### Weekend / holiday gap protection

Before an IG FR40 close that precedes a non-trading day (weekend or Euronext full closure), the live cycle **directionally flattens** the book by hedging — it does **not** scratch losing legs.

- **Calendar:** local Euronext list only (1 Jan, Good Friday, Easter Monday, 1 May, 25–26 Dec). Bastille Day (14 Jul) and similar French bank holidays stay **open** on Euronext — no flatten. No holiday API; the LLM is not trusted to know closures.
- **Window:** default `market_close_paris=22:00`, `flatten_lead_minutes=30` (so 21:30 + 21:45 slots). Config: `flatten_before_close`, `flatten_lead_minutes`, `market_close_paris`.
- **Sync:** Paper runs OHLC `process_bar` fills on each new 15m candle. **Live never fills from candles** — IG is source of truth. Live polls `GET /workingorders` every cycle; `GET /positions` is event-driven (arm, WO change, after mirror, flatten, or once per new 15m bar). Opens/closes bind by IG `dealId`. Desync triggers replace-rebuild of the open book (and quarantines phantom local closes). Dashboard **Sync book from IG** rebuilds the open book the same way. Unresolved desync skips the LLM.
- **Enforcement:** prompt gets `market_clock.flatten_now`; if the LLM fails or skips, code `open_market_position` (real `POST /positions/otc` in Live) with size=`|net|` and cancels resting `entry` orders. Monday leaves a `Covered` book for normal profit-only management.

Config lives under `data/cac40/{slug}/live/` (`live_config.json`, `status.json`, `state.json`, `decisions_log.json`, `journal/{cycle}/cycle.json` + charts).

```bash
# Live worker
./sail up -d worker-cac40-live
./sail exec worker-cac40-live python -m chatbot.interfaces.worker_cac40_live --once
./sail logs worker-cac40-live
```

Poll interval: `CAC40_LIVE_POLL_SECONDS` (default 60). Bot cycles align to 15m candle closes in Europe/Paris at `:00:15`, `:15:15`, `:30:15`, `:45:15` (15s after the bar closes so OHLC is available). With **Fixed rate** LLM schedule, Gemini spacing uses a persisted wall-clock gate (`live/llm_schedule.json`, seeded from the last LLM cycle) so restarts / Run cycle now do not reset the 6h (or configured) interval.

**Live OHLC & IG historical allowance:** dashboard Paper/Live cycles use the local per-bot CSV (`data/cac40/{slug}/ohlc/…`), top up missing 15m bars from IG, then resample 1H/1D locally for triggers/charts/LLM. Small lags use a cheap `max≈8` fetch **only if those bars connect** to the last CSV candle; otherwise (or after a multi-day outage) the cycle **range-fills from last candle → now** so mid-session holes are never spliced into the CSV (overnight/weekend breaks are allowed). If the chart lookback still has a mid-session gap, the LLM is **skipped**. A one-time **Upload** or **Sync from IG** is required before arming. Fixed-rate LLM spacing alone does **not** reduce `/prices` spend — this local cache does. While a bot is armed (`paper`/`live`), `worker-cac40-ohlc` skips that slug. If top-up hits a 403 but the CSV is still recent (≤2×15m), the cycle continues with a `stale_data` warning; older cache **skips the LLM**. The OHLC / Live panels show last-known `remaining / total` points and reset countdown from IG `metadata.allowance` (updated only on successful `/prices` responses).

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
- Background worker (`worker-cac40-ohlc`, poll `CAC40_OHLC_POLL_SECONDS` default 900) tops up tenants with active CAC40 + IG + existing CSV (no cron bootstrap of empty files). Skips bots armed for Paper/Live (their live worker tops up instead).

The OHLC card shows **last candle**, **age in hours**, **mid-session gap count** (with fix steps: Sync and/or re-upload CSV), last sync (manual/worker), IG historical allowance, fetch window, and worker cycle time. Overnight/weekend breaks are ignored.

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

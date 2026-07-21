# CAC40 mean-reversion bot (Evenor)

Hedge-mode CAC40 bot: IG connectors, multimodal Gemini decisions, HedgeLedger backtest, Fund Manager heartbeat (`source=evenor`).

## Layout

| Path | Role |
|------|------|
| `src/chatbot/cac40/` | Core engine (IG, charts, LLM, RiskGate, HedgeLedger, backtest, live scheduler) |
| `prompts/cac40/system.md` | LLM system prompt |
| `/dashboard/bots/{slug}/cac40` | Backtest UI integration |
| `/dashboard/bots/{slug}/cac40/runs/{id}` | Run report + LLM decision browser (charts sent to Gemini) |
| Connectors type `ig` | IG credentials in/out |
| Integration type `cac40_backtest` | FM URL/token + defaults |

## CLI

```bash
# Backtest (replay = no Gemini calls unless cache filled)
PYTHONPATH=src python -m chatbot cac40 backtest path/to/cac40_15m.csv --llm-mode replay

# Live / demo single cycle (dry-run default)
PYTHONPATH=src python -m chatbot cac40 live --once --dry-run
```

## Fund Manager

Heartbeat: `POST /jessebot/notify-up` with `source=evenor` and optional `X-Notify-Token`.
UI: **Jesse & Evenor** menu — expandable multi-leg PnL for Evenor rows.

## Data

### Free (built-in UI)

On **CAC40 Backtest** click **Fetch from Yahoo Finance (^FCHI)**:
- Source: Yahoo Finance cash index `^FCHI`
- Interval: 15m, last ~60 days (~2000 bars)
- No account required

### Longer history

Yahoo caps 15m at 60 days. For multi-year backtests, buy a futures CSV (e.g. [backtestmarket CAC MX 15m](https://www.backtestmarket.com/en/cac-40-15m)) and use **Upload CSV** with source **BacktestMarket (MX 15m)** — Evenor converts `;` / no-header files automatically.

You can also place a file at `data/cac40/{slug}/ohlc/cac40_15m.csv` (columns `Date,Open,High,Low,Close[,Volume]`).

Note: Yahoo = cash index; live trading = IG CFD — calibrate `spread_points`.

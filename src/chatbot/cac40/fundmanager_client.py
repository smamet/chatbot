from __future__ import annotations

import logging
from typing import Any

import httpx

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger

logger = logging.getLogger(__name__)


class FundManagerClient:
    """POST heartbeat to Fund Manager Jesse notify-up with source=evenor."""

    def __init__(self, config: Cac40Config) -> None:
        self.config = config

    def notify(self, ledger: HedgeLedger, *, error: str | None = None) -> bool:
        if not self.config.fundmanager_url:
            logger.debug("Fund Manager URL not configured; skip notify")
            return False
        snap = ledger.get_snapshot()
        pnl = ledger.pnl_payload()
        payload: dict[str, Any] = {
            "id": self.config.bot_id,
            "source": "evenor",
            "strategy": self.config.strategy_name,
            "symbol": self.config.symbol,
            "exchange": "IG",
            "timeframe": self.config.timeframe,
            "last_price": snap.last_price,
            "position": ledger.position_summary(),
            "position_pnl": pnl["net_upl"],
            "position_qty": pnl["legs_count"],
            "mode": self.config.ig_acc_type.lower(),
            "phase": snap.phase,
            "pnl": pnl,
            "positions": [p.to_dict() for p in snap.positions],
            "working_orders": [o.to_dict() for o in snap.working_orders],
            "last_levels": snap.last_levels,
            "healthy": error is None,
            "error": error,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.fundmanager_token:
            headers["X-Notify-Token"] = self.config.fundmanager_token
        url = self.config.fundmanager_url.rstrip("/")
        if not url.endswith("notify-up"):
            url = f"{url}/jessebot/notify-up"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=20.0)
            if resp.status_code >= 400:
                logger.error("FM notify failed %s: %s", resp.status_code, resp.text)
                return False
            return True
        except Exception:
            logger.exception("FM notify error")
            return False

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote

import httpx

from evenor.adapters.quickbooks.oauth import refresh_access_token

logger = logging.getLogger(__name__)

_SANDBOX_BASE = "https://sandbox-quickbooks.api.intuit.com"
_PRODUCTION_BASE = "https://quickbooks.api.intuit.com"


class QuickBooksError(Exception):
    pass


class QuickBooksClient:
    """QuickBooks Online query API with token refresh."""

    def __init__(self, config: dict[str, Any], *, timeout: float = 20.0) -> None:
        self._config = dict(config)
        self._timeout = timeout
        self._environment = str(config.get("environment", "sandbox")).strip().lower()
        self._base_url = _PRODUCTION_BASE if self._environment == "production" else _SANDBOX_BASE
        self._last_customer_id: str | None = None

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def _realm_id(self) -> str:
        realm = str(self._config.get("realm_id", "")).strip()
        if not realm:
            raise QuickBooksError("QuickBooks is not connected (missing realm_id)")
        return realm

    def _ensure_access_token(self) -> str:
        access = str(self._config.get("access_token", "")).strip()
        refresh = str(self._config.get("refresh_token", "")).strip()
        client_id = str(self._config.get("client_id", "")).strip()
        client_secret = str(self._config.get("client_secret", "")).strip()
        expires_at = int(self._config.get("token_expires_at") or 0)
        if not refresh or not client_id or not client_secret:
            raise QuickBooksError("QuickBooks OAuth credentials are incomplete")
        if access and expires_at > int(time.time()) + 30:
            return access
        tokens = refresh_access_token(
            refresh_token=refresh,
            client_id=client_id,
            client_secret=client_secret,
        )
        self._config["access_token"] = tokens.access_token
        self._config["refresh_token"] = tokens.refresh_token
        self._config["token_expires_at"] = tokens.expires_at
        return tokens.access_token

    def _query(self, sql: str) -> list[dict[str, Any]]:
        token = self._ensure_access_token()
        realm_id = self._realm_id()
        url = f"{self._base_url}/v3/company/{realm_id}/query"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    url,
                    params={"query": sql, "minorversion": "65"},
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise QuickBooksError(f"QuickBooks query failed: {exc}") from exc
        query_response = payload.get("QueryResponse") if isinstance(payload, dict) else None
        if not isinstance(query_response, dict):
            return []
        for key in ("Customer", "Invoice", "Estimate"):
            rows = query_response.get(key)
            if isinstance(rows, list):
                return rows
        return []

    @staticmethod
    def _escape(value: str) -> str:
        return value.replace("'", "\\'")

    def ping(self) -> None:
        self._query("SELECT Id FROM Customer MAXRESULTS 1")

    def find_customer(self, *, email: str | None = None, phone: str | None = None) -> str | None:
        _ = phone
        self._last_customer_id = None
        if not email:
            return None
        safe_email = self._escape(email.strip().lower())
        rows = self._query(
            "SELECT Id, DisplayName, PrimaryEmailAddr FROM Customer "
            f"WHERE PrimaryEmailAddr = '{safe_email}' MAXRESULTS 1"
        )
        if not rows:
            return None
        row = rows[0]
        self._last_customer_id = str(row.get("Id", "")).strip() or None
        display = str(row.get("DisplayName") or self._last_customer_id or "").strip()
        return display or None

    def get_orders(self, customer: str, limit: int) -> list[dict[str, Any]]:
        customer_id = self._last_customer_id or customer
        safe_id = self._escape(customer_id)
        rows = self._query(
            "SELECT Id, DocNumber, TxnDate, TotalAmt, Balance, EmailStatus FROM Invoice "
            f"WHERE CustomerRef = '{safe_id}' ORDERBY TxnDate DESC MAXRESULTS {max(1, limit)}"
        )
        return [_normalize_invoice(row) for row in rows]

    def get_quotations(self, customer: str, limit: int) -> list[dict[str, Any]]:
        customer_id = self._last_customer_id or customer
        safe_id = self._escape(customer_id)
        rows = self._query(
            "SELECT Id, DocNumber, TxnDate, TotalAmt, EmailStatus, ExpirationDate FROM Estimate "
            f"WHERE CustomerRef = '{safe_id}' ORDERBY TxnDate DESC MAXRESULTS {max(1, limit)}"
        )
        return [_normalize_estimate(row) for row in rows]

    def get_customer_profile(self, customer: str) -> dict[str, Any]:
        return {"name": customer}

    def get_matched_contact(
        self,
        *,
        email: str | None = None,
        phone: str | None = None,
        customer: str,
    ) -> dict[str, Any] | None:
        _ = phone, customer
        if not email:
            return None
        return {"email": email.strip().lower()}

    def get_current_item_prices(
        self, customer: str, item_codes: list[str]
    ) -> dict[str, dict[str, Any]]:
        _ = customer, item_codes
        return {}


def _normalize_invoice(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("DocNumber") or row.get("Id"),
        "transaction_date": row.get("TxnDate"),
        "status": row.get("EmailStatus") or "Invoice",
        "grand_total": row.get("TotalAmt"),
        "balance": row.get("Balance"),
    }


def _normalize_estimate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("DocNumber") or row.get("Id"),
        "transaction_date": row.get("TxnDate"),
        "status": row.get("EmailStatus") or "Estimate",
        "grand_total": row.get("TotalAmt"),
        "valid_till": row.get("ExpirationDate"),
    }


def build_quickbooks_client(config: dict[str, Any]) -> QuickBooksClient:
    return QuickBooksClient(config)

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rapidfuzz import fuzz

from evenor.adapters.erpnext.client import ErpNextClient


class LineMatchStatus(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class ItemCandidate:
    item_code: str
    item_name: str
    rate: float | None
    uom: str | None
    score: float


@dataclass(frozen=True, slots=True)
class ResolvedLine:
    requested_label: str
    qty: int
    item_code: str | None
    item_name: str | None
    rate: float | None
    uom: str | None
    match_score: float
    status: LineMatchStatus
    candidates: tuple[ItemCandidate, ...]


def normalize_product_key(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    lowered = lowered.lower()
    return re.sub(r"[^a-z0-9]+", "", lowered)


class ProductResolver:
    AUTO_SCORE = 90.0
    AMBIGUOUS_SCORE = 70.0
    SCORE_GAP = 10.0

    def __init__(self, client: ErpNextClient) -> None:
        self._client = client

    def resolve_line(
        self,
        *,
        product: str,
        qty: int,
        item_code: str | None = None,
    ) -> ResolvedLine:
        label = product.strip()
        if item_code:
            row = self._client.get_item_by_code(item_code)
            if row:
                return self._resolved_from_row(label, qty, row, score=100.0)
        if label:
            row = self._client.get_item_by_code(label)
            if row:
                return self._resolved_from_row(label, qty, row, score=100.0)
        norm_label = normalize_product_key(label)
        if norm_label:
            for token in self._search_tokens(label):
                for row in self._client.search_items(token, limit=30):
                    code = str(row.get("item_code", ""))
                    name = str(row.get("item_name", ""))
                    if normalize_product_key(code) == norm_label or normalize_product_key(name) == norm_label:
                        return self._resolved_from_row(label, qty, row, score=98.0)
        candidates = self._rank_candidates(label)
        if not candidates:
            return ResolvedLine(
                requested_label=label,
                qty=qty,
                item_code=None,
                item_name=None,
                rate=None,
                uom=None,
                match_score=0.0,
                status=LineMatchStatus.NOT_FOUND,
                candidates=(),
            )
        top = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else 0.0
        if top.score >= self.AUTO_SCORE and (top.score - second_score) >= self.SCORE_GAP:
            return ResolvedLine(
                requested_label=label,
                qty=qty,
                item_code=top.item_code,
                item_name=top.item_name,
                rate=top.rate,
                uom=top.uom,
                match_score=top.score,
                status=LineMatchStatus.RESOLVED,
                candidates=candidates,
            )
        strong = [c for c in candidates if c.score >= self.AMBIGUOUS_SCORE]
        if len(strong) >= 2:
            status = LineMatchStatus.AMBIGUOUS
        elif top.score >= self.AMBIGUOUS_SCORE:
            status = LineMatchStatus.RESOLVED
            return ResolvedLine(
                requested_label=label,
                qty=qty,
                item_code=top.item_code,
                item_name=top.item_name,
                rate=top.rate,
                uom=top.uom,
                match_score=top.score,
                status=status,
                candidates=candidates,
            )
        return ResolvedLine(
            requested_label=label,
            qty=qty,
            item_code=None,
            item_name=None,
            rate=None,
            uom=None,
            match_score=top.score,
            status=LineMatchStatus.NOT_FOUND,
            candidates=candidates,
        )

    def resolve_all(
        self,
        lines: list[dict[str, Any]],
    ) -> list[ResolvedLine]:
        out: list[ResolvedLine] = []
        for raw in lines:
            product = str(raw.get("product", "")).strip()
            try:
                qty = int(raw.get("qty", 0))
            except (TypeError, ValueError):
                qty = 0
            item_code = str(raw.get("item_code", "")).strip() or None
            if not product or qty <= 0:
                continue
            out.append(
                self.resolve_line(product=product, qty=qty, item_code=item_code)
            )
        return out

    def _search_tokens(self, label: str) -> list[str]:
        parts = re.split(r"\s+", label.strip())
        tokens = [label] + [p for p in parts if len(p) >= 2]
        seen: set[str] = set()
        out: list[str] = []
        for token in tokens:
            key = token.lower()
            if key not in seen:
                seen.add(key)
                out.append(token)
        return out

    def _rank_candidates(self, label: str) -> tuple[ItemCandidate, ...]:
        norm_label = normalize_product_key(label)
        if not norm_label:
            return ()
        merged: dict[str, dict[str, Any]] = {}
        for token in self._search_tokens(label):
            for row in self._client.search_items(token, limit=25):
                code = str(row.get("item_code", "")).strip()
                if code:
                    merged[code] = row
        scored: list[ItemCandidate] = []
        for row in merged.values():
            code = str(row.get("item_code", ""))
            name = str(row.get("item_name", ""))
            score = max(
                fuzz.ratio(norm_label, normalize_product_key(code)),
                fuzz.ratio(norm_label, normalize_product_key(name)),
                fuzz.partial_ratio(norm_label, normalize_product_key(name)),
            )
            scored.append(
                ItemCandidate(
                    item_code=code,
                    item_name=name,
                    rate=_to_float(row.get("standard_rate")),
                    uom=str(row.get("stock_uom") or "") or None,
                    score=float(score),
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return tuple(scored[:10])

    @staticmethod
    def _resolved_from_row(
        label: str,
        qty: int,
        row: dict[str, Any],
        *,
        score: float,
    ) -> ResolvedLine:
        code = str(row.get("item_code", ""))
        name = str(row.get("item_name", ""))
        candidate = ItemCandidate(
            item_code=code,
            item_name=name,
            rate=_to_float(row.get("standard_rate")),
            uom=str(row.get("stock_uom") or "") or None,
            score=score,
        )
        return ResolvedLine(
            requested_label=label,
            qty=qty,
            item_code=code,
            item_name=name,
            rate=candidate.rate,
            uom=candidate.uom,
            match_score=score,
            status=LineMatchStatus.RESOLVED,
            candidates=(candidate,),
        )


def resolved_lines_to_json(lines: list[ResolvedLine]) -> str:
    import json

    payload = [
        {
            "requested_label": line.requested_label,
            "qty": line.qty,
            "item_code": line.item_code,
            "item_name": line.item_name,
            "rate": line.rate,
            "uom": line.uom,
            "match_score": line.match_score,
            "status": line.status.value,
            "candidates": [
                {
                    "item_code": c.item_code,
                    "item_name": c.item_name,
                    "rate": c.rate,
                    "uom": c.uom,
                    "score": c.score,
                }
                for c in line.candidates
            ],
        }
        for line in lines
    ]
    return json.dumps(payload, ensure_ascii=True)


def resolved_lines_from_json(raw: str | None) -> list[dict[str, Any]]:
    import json

    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

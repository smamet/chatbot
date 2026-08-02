"""Resolve human tickers (EURUSD, CAC40) to TRADEABLE IG epics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chatbot.trader.profiles import PROFILES, get_profile


# Offline seed when IG search is unavailable.
_ALIAS_EPICS: dict[str, str] = {
    "EURUSD": "CS.D.EURUSD.MINI.IP",
    "EUR/USD": "CS.D.EURUSD.MINI.IP",
    "CAC40": "IX.D.CAC.BMU.IP",
    "CAC": "IX.D.CAC.BMU.IP",
    "FR40": "IX.D.CAC.BMU.IP",
    "FRANCE40": "IX.D.CAC.BMU.IP",
    "DAX": "IX.D.DAX.IFMM.IP",
    "GER40": "IX.D.DAX.IFMM.IP",
}


@dataclass(frozen=True)
class ResolvedEpic:
    epic: str
    name: str = ""
    symbol: str = ""
    source: str = "search"  # search | explicit | alias | profile


def looks_like_ig_epic(value: str | None) -> bool:
    """True for IG-shaped epics (``CS.D.EURUSD.MINI.IP``), not bare tickers."""
    text = str(value or "").strip().upper()
    if not text or "." not in text:
        return False
    parts = text.split(".")
    return len(parts) >= 3 and parts[0] in {"CS", "IX", "KA", "HK", "ED", "CC", "MW", "DO", "EF"}


def search_terms_for_symbol(symbol: str) -> list[str]:
    raw = str(symbol or "").strip()
    if not raw:
        return []
    terms = [raw]
    compact = raw.replace(" ", "").replace("/", "").upper()
    if compact and compact != raw.upper():
        terms.append(compact)
    if len(compact) == 6 and compact.isalpha():
        # EURUSD → EUR/USD
        pair = f"{compact[:3]}/{compact[3:]}"
        if pair not in terms:
            terms.append(pair)
        spaced = f"{compact[:3]} {compact[3:]}"
        if spaced not in terms:
            terms.append(spaced)
    # Dedupe preserving order
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


def alias_epic_for_symbol(symbol: str | None, *, profile_id: str | None = None) -> str | None:
    key = str(symbol or "").strip().upper().replace(" ", "")
    if key in _ALIAS_EPICS:
        return _ALIAS_EPICS[key]
    slash = str(symbol or "").strip().upper()
    if slash in _ALIAS_EPICS:
        return _ALIAS_EPICS[slash]
    if profile_id:
        return get_profile(profile_id).default_epic
    for profile in PROFILES.values():
        if profile.default_symbol.upper().replace(" ", "") == key:
            return profile.default_epic
    return None


def _epic_family_hint(epic: str) -> str:
    """Cheap CFD vs spread-bet guess from epic string only (no GET /markets)."""
    ep = (epic or "").strip().upper()
    parts = ep.split(".")
    tag = parts[3] if len(parts) >= 4 else ""
    if tag in ("CFS", "IFS", "IDF", "IFA", "CASH", "CFD", "MINI", "BMU") or any(
        t in ep for t in (".CFS.", ".IFS.", ".IDF.", ".CASH.", ".CFD.", ".MINI.", ".BMU.")
    ):
        return "CFD"
    if tag in ("TODAY", "DAILY") or ".TODAY." in ep:
        # Ambiguous (CAC DAILY can be CFD); don't force spread-bet.
        return "UNKNOWN"
    return "UNKNOWN"


def _rank_market_row(
    row: dict[str, Any],
    *,
    symbol: str,
    account_type: str,
    require_tradeable: bool = True,
    epic_product_hint: Any | None = None,
    epic_compatible: Any | None = None,
) -> int | None:
    """Lower score is better. None = reject."""
    epic = str(row.get("epic") or "").strip()
    status = str(row.get("marketStatus") or "").upper()
    if not epic:
        return None
    if require_tradeable and status and status != "TRADEABLE":
        return None
    # Autocomplete: drop only clearly unusable statuses (keep CLOSED/OFFLINE for config).
    if status in {"UNAVAILABLE", "SUSPENDED"}:
        return None
    name = str(row.get("instrumentName") or row.get("instrument") or epic)
    upper = f"{epic} {name}".upper()
    if any(x in upper for x in ("BULL", "BEAR", "WEEKEND", " KO", "KNOCK")):
        return None

    score = 100
    sym = str(symbol or "").strip().upper().replace(" ", "").replace("/", "")
    name_compact = name.upper().replace(" ", "").replace("/", "")
    epic_u = epic.upper()

    if sym and sym in name_compact:
        score -= 40
    if sym and sym in epic_u.replace(".", ""):
        score -= 25
    if status == "TRADEABLE":
        score -= 5

    # Prefer Mini / Cash CFD-style epics for small size.
    if ".MINI." in epic_u or epic_u.endswith(".MINI.IP"):
        score -= 30
    if "BMU" in epic_u or ".CASH." in epic_u:
        score -= 20
    if "IFMM" in epic_u or "IFA" in epic_u or "IFS" in epic_u:
        score -= 10
    if "ITA" in upper or epic_u.startswith("EF.D."):
        score += 40  # avoid Italian/knockout-ish variants

    acc = (account_type or "").strip().upper()
    # Prefer epic-string heuristics for search ranking — calling IG get_market
    # per row is slow and defeats autocomplete caching.
    family = _epic_family_hint(epic)
    if callable(epic_product_hint):
        try:
            hinted = str(epic_product_hint(epic) or "").upper()
            if hinted in {"CFD", "SPREADBET"}:
                family = hinted
        except Exception:
            pass
    if acc == "CFD" and family == "SPREADBET":
        score += 50
    elif acc == "SPREADBET" and family == "CFD":
        score += 50
    elif acc == "CFD" and family == "CFD":
        score -= 15

    if callable(epic_compatible):
        try:
            if not epic_compatible(epic=epic, account_type=acc or None):
                score += 50
        except Exception:
            pass

    return score


def rank_search_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str,
    account_type: str = "",
    require_tradeable: bool = True,
    epic_product_hint: Any | None = None,
    epic_compatible: Any | None = None,
) -> list[tuple[int, str, str]]:
    ranked: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        score = _rank_market_row(
            row,
            symbol=symbol,
            account_type=account_type,
            require_tradeable=require_tradeable,
            epic_product_hint=epic_product_hint,
            epic_compatible=epic_compatible,
        )
        if score is None:
            continue
        epic = str(row.get("epic") or "").strip()
        if not epic or epic in seen:
            continue
        seen.add(epic)
        name = str(row.get("instrumentName") or row.get("instrument") or epic)
        ranked.append((score, epic, name))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return ranked


def resolve_ticker_to_epic(
    ig: Any | None,
    symbol: str,
    *,
    explicit_epic: str | None = None,
    profile_id: str | None = None,
    account_type: str | None = None,
) -> ResolvedEpic | None:
    """
    Map a human symbol to a TRADEABLE IG epic.

    Preference order:
    1. Explicit full IG epic (validated when ``ig`` available)
    2. IG ``search_markets`` ranked for Mini/Cash + account type
    3. Offline alias / profile default
    """
    sym = str(symbol or "").strip()
    explicit = str(explicit_epic or "").strip()

    if looks_like_ig_epic(explicit):
        if ig is not None:
            try:
                market = ig.get_market(explicit)
                instrument = (market.get("instrument") or {}) if isinstance(market, dict) else {}
                name = str(instrument.get("name") or explicit)
                return ResolvedEpic(epic=explicit, name=name, symbol=sym, source="explicit")
            except Exception:
                pass
        return ResolvedEpic(epic=explicit, name=explicit, symbol=sym, source="explicit")

    # Bare ticker pasted into epic field — treat as symbol.
    search_symbol = sym or (explicit if explicit and not looks_like_ig_epic(explicit) else "")
    if not search_symbol:
        return None

    acc = (account_type or "").strip().upper()
    if ig is not None and not acc:
        try:
            acc = str(ig.resolve_account_type() or "").strip().upper()
        except Exception:
            acc = ""

    if ig is not None:
        rows: list[dict[str, Any]] = []
        for term in search_terms_for_symbol(search_symbol):
            try:
                found = ig.search_markets(term) or []
            except Exception:
                found = []
            for row in found:
                if isinstance(row, dict):
                    rows.append(row)
        hint_fn = getattr(ig, "epic_product_hint", None)
        compat_fn = getattr(ig, "epic_compatible_with_account", None)
        ranked = rank_search_rows(
            rows,
            symbol=search_symbol,
            account_type=acc,
            epic_product_hint=hint_fn if callable(hint_fn) else None,
            epic_compatible=compat_fn if callable(compat_fn) else None,
        )
        if ranked:
            _score, epic, name = ranked[0]
            return ResolvedEpic(epic=epic, name=name, symbol=search_symbol, source="search")

    alias = alias_epic_for_symbol(search_symbol, profile_id=profile_id)
    if alias:
        source = "profile" if profile_id and alias == get_profile(profile_id).default_epic else "alias"
        return ResolvedEpic(epic=alias, name=alias, symbol=search_symbol, source=source)
    return None


def autocomplete_symbol_rows(
    ig: Any,
    query: str,
    *,
    account_type: str | None = None,
    limit: int = 12,
    cache_dir: Any | None = None,
    ttl_seconds: int = 24 * 60 * 60,
) -> dict[str, Any]:
    """
    Autocomplete payload: ``{results, ig_calls, cache_hits}``.

    Uses a single IG search term (not every alias) and a 24h disk cache so
    typing does not re-query IG for the same prefix.
    """
    from pathlib import Path

    from chatbot.trader.ig_market_search_cache import cached_search_markets

    q = str(query or "").strip()
    if not q or ig is None:
        return {"results": [], "ig_calls": 0, "cache_hits": 0}
    acc = (account_type or "").strip().upper()
    if not acc:
        try:
            acc = str(ig.resolve_account_type() or "").strip().upper()
        except Exception:
            acc = ""
    # One primary term for autocomplete; aliases only if primary returns nothing.
    terms = [q]
    for alt in search_terms_for_symbol(q):
        if alt not in terms:
            terms.append(alt)

    rows: list[dict[str, Any]] = []
    ig_calls = 0
    cache_hits = 0
    cdir = Path(cache_dir) if cache_dir else None
    for term in terms:
        found, from_cache = cached_search_markets(
            ig,
            term,
            cache_dir=cdir,
            acc_type=acc,
            ttl_seconds=ttl_seconds,
        )
        if from_cache:
            cache_hits += 1
        else:
            ig_calls += 1
        rows.extend(found)
        if rows:
            break  # stop after first productive term

    ranked = rank_search_rows(
        rows,
        symbol=q,
        account_type=acc,
        require_tradeable=False,
        # Avoid per-row GET /markets during autocomplete.
        epic_product_hint=None,
        epic_compatible=None,
    )
    out: list[dict[str, str]] = []
    for _score, epic, name in ranked[: max(1, int(limit))]:
        label = q.upper().replace(" ", "") if len(q) <= 12 else name
        out.append({"symbol": label, "epic": epic, "name": name})
    return {"results": out, "ig_calls": ig_calls, "cache_hits": cache_hits}

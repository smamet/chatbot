"""Normalize live/backtest cycle payloads into compact UI ops-log sections."""

from __future__ import annotations

import re
from typing import Any

_ERROR_CODE_RE = re.compile(r"IG errorCode:\s*(\S+)", re.I)
_HTTP_RE = re.compile(r"HTTP\s+(\d{3})", re.I)
_DEAL_STATUS_RE = re.compile(r"dealStatus=([A-Za-z_]+)", re.I)


def build_cycle_ops_log(entry: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Build collapsible ops-log sections for a decision/cycle row.

    Each section: ``{"title": str, "lines": list[str]}``. Empty sections omitted.
    """
    if not isinstance(entry, dict):
        return []
    sections: list[dict[str, Any]] = []
    for title, lines in (
        ("Trigger", _trigger_lines(entry)),
        ("RiskGate", _gate_lines(entry)),
        ("Mirror / IG", _mirror_lines(entry)),
        ("Book sync", _book_sync_lines(entry)),
        ("OHLC / flatten", _ohlc_flatten_lines(entry)),
        ("Fills", _fill_lines(entry)),
    ):
        if lines:
            sections.append({"title": title, "lines": lines})
    return sections


def ops_log_line_count(sections: list[dict[str, Any]] | None) -> int:
    if not sections:
        return 0
    return sum(len(s.get("lines") or []) for s in sections)


def _trigger_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if entry.get("skipped"):
        lines.append("skipped=true (LLM not required this cycle)")
    if entry.get("dry_run"):
        lines.append("dry_run=true")
    mode = entry.get("llm_mode")
    if mode:
        lines.append(f"llm_mode={mode}")
    trigger = entry.get("llm_trigger")
    if isinstance(trigger, dict) and trigger:
        bits = [f"{k}={trigger[k]}" for k in sorted(trigger) if trigger[k] not in (None, "", [])]
        if bits:
            lines.append("llm_trigger " + " ".join(bits))
    elif trigger not in (None, "", {}):
        lines.append(f"llm_trigger={trigger}")
    err = entry.get("llm_error")
    if err:
        lines.append(f"llm_error={_trim(str(err))}")
    return lines


def _gate_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in entry.get("executed") or []:
        lines.append(f"OK {_trim(str(item))}")
    for item in entry.get("rejected") or []:
        lines.append(f"REJECT {_trim(str(item))}")
    return lines


def _mirror_lines(entry: dict[str, Any]) -> list[str]:
    mirror = entry.get("mirror")
    if not isinstance(mirror, list) or not mirror:
        return []
    lines: list[str] = []
    for row in mirror:
        if not isinstance(row, dict):
            continue
        cid = row.get("connector_id", "?")
        prefix = f"conn#{cid}"
        for p in row.get("placed") or []:
            lines.append(f"{prefix} PLACE {_format_mirror_item(p)}")
        for c in row.get("cancelled") or []:
            lines.append(f"{prefix} CANCEL {_format_mirror_item(c)}")
        for a in row.get("amended") or []:
            lines.append(f"{prefix} AMEND {_format_mirror_item(a)}")
        for d in row.get("deferred") or []:
            if isinstance(d, dict):
                lines.append(f"{prefix} DEFER {_format_mirror_item(d)}")
            else:
                lines.append(f"{prefix} DEFER {_trim(str(d))}")
        for err in row.get("errors") or []:
            lines.append(f"{prefix} ERROR {_format_mirror_error(err)}")
        if not any(row.get(k) for k in ("placed", "cancelled", "amended", "deferred", "errors")):
            lines.append(f"{prefix} no-op")
    return lines


def _format_mirror_item(item: Any) -> str:
    if not isinstance(item, dict):
        return _trim(str(item))
    parts: list[str] = []
    oid = item.get("order_id")
    if oid:
        parts.append(str(oid))
    did = item.get("deal_id")
    if did:
        parts.append(f"deal={did}")
    via = item.get("via")
    if via:
        parts.append(f"via={via}")
    level = item.get("level")
    if level is not None:
        parts.append(f"level={level}")
    status = item.get("deal_status") or item.get("dealStatus")
    if status:
        parts.append(f"status={status}")
    reason = item.get("reason")
    if reason:
        parts.append(f"reason={_trim(str(reason), 80)}")
    ref = item.get("deal_reference") or item.get("dealReference")
    if ref:
        parts.append(f"ref={ref}")
    return " ".join(parts) if parts else _trim(str(item))


def _format_mirror_error(err: Any) -> str:
    if isinstance(err, dict):
        parts: list[str] = []
        if err.get("order_id"):
            parts.append(str(err["order_id"]))
        code = err.get("error_code") or err.get("errorCode")
        if code:
            parts.append(f"errorCode={code}")
        http = err.get("http_status") or err.get("status")
        if http:
            parts.append(f"HTTP {http}")
        msg = err.get("error") or err.get("message") or ""
        if msg:
            parts.append(_compact_error_text(str(msg)))
        return " ".join(parts) if parts else _trim(str(err))
    return _compact_error_text(str(err))


def _compact_error_text(text: str) -> str:
    text = text.strip()
    code_m = _ERROR_CODE_RE.search(text)
    http_m = _HTTP_RE.search(text)
    status_m = _DEAL_STATUS_RE.search(text)
    bits: list[str] = []
    if http_m:
        bits.append(f"HTTP {http_m.group(1)}")
    if code_m:
        bits.append(f"errorCode={code_m.group(1)}")
    if status_m:
        bits.append(f"dealStatus={status_m.group(1)}")
    if bits:
        # Keep a short head for context (action name etc.).
        head = text.split("\n", 1)[0]
        head = _trim(head, 100)
        return f"{head} [{', '.join(bits)}]" if head else ", ".join(bits)
    return _trim(text, 180)


def _book_sync_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key, label in (
        ("reconcile", "reconcile"),
        ("working_order_sync", "wo_sync"),
        ("book_repair", "book_repair"),
    ):
        payload = entry.get(key)
        if not isinstance(payload, dict) or not payload:
            continue
        summary = _summarize_sync_dict(payload)
        if summary:
            lines.append(f"{label} {summary}")
    return lines


def _summarize_sync_dict(payload: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in (
        "ran",
        "repaired",
        "desync",
        "changed",
        "ig_net",
        "local_net",
        "mode",
    ):
        if key in payload and payload[key] not in (None, "", []):
            bits.append(f"{key}={payload[key]}")
    for key in ("closed", "opened", "imported", "dropped", "quarantined", "warnings"):
        val = payload.get(key)
        if isinstance(val, list) and val:
            bits.append(f"{key}={len(val)}")
            if key == "warnings":
                for w in val[:3]:
                    bits.append(f"warn={_trim(str(w), 80)}")
        elif val not in (None, "", [], False) and not isinstance(val, list):
            bits.append(f"{key}={val}")
    repair = payload.get("repair")
    if isinstance(repair, dict) and repair:
        bits.append("repair=" + _summarize_sync_dict(repair))
    return " ".join(bits)


def _ohlc_flatten_lines(entry: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    feed = entry.get("ohlc_feed")
    if isinstance(feed, dict) and feed:
        bits = []
        for key in ("stale", "skip", "error", "source"):
            if feed.get(key) not in (None, "", False):
                bits.append(f"{key}={feed[key]}")
        warnings = feed.get("warnings") or feed.get("warning")
        if isinstance(warnings, list) and warnings:
            bits.append(f"warnings={len(warnings)}")
            for w in warnings[:3]:
                bits.append(f"warn={_trim(str(w), 80)}")
        elif warnings:
            bits.append(f"warn={_trim(str(warnings), 80)}")
        allowance = feed.get("allowance")
        if isinstance(allowance, dict) and allowance:
            rem = allowance.get("remainingAllowance")
            tot = allowance.get("totalAllowance")
            if rem is not None or tot is not None:
                bits.append(f"allowance={rem}/{tot}")
        if bits:
            lines.append("ohlc " + " ".join(bits))
    flat = entry.get("auto_flatten")
    if isinstance(flat, dict) and flat:
        bits = [
            f"{k}={flat[k]}"
            for k in ("ran", "size", "net_before", "net_after", "side")
            if flat.get(k) not in (None, "")
        ]
        errors = flat.get("errors") or []
        if errors:
            bits.append(f"errors={len(errors)}")
            for e in errors[:3]:
                bits.append(f"err={_trim(str(e), 80)}")
        if bits:
            lines.append("auto_flatten " + " ".join(bits))
    clock = entry.get("market_clock")
    if isinstance(clock, dict) and clock.get("flatten_now"):
        lines.append(
            "market_clock flatten_now=true "
            f"net={clock.get('net_exposure', clock.get('local_net', '—'))}"
        )
    return lines


def _fill_lines(entry: dict[str, Any]) -> list[str]:
    fills = entry.get("fill_events")
    if not isinstance(fills, list) or not fills:
        return []
    lines = [f"count={len(fills)}"]
    for ev in fills[:8]:
        if isinstance(ev, dict):
            bits = [
                str(ev.get("type") or ev.get("event") or "fill"),
            ]
            for key in ("id", "order_id", "position_id", "side", "size", "price", "level"):
                if ev.get(key) not in (None, ""):
                    bits.append(f"{key}={ev[key]}")
            lines.append(" ".join(bits))
        else:
            lines.append(_trim(str(ev)))
    if len(fills) > 8:
        lines.append(f"… +{len(fills) - 8} more")
    return lines


def _trim(text: str, max_len: int = 160) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

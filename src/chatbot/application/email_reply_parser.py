from __future__ import annotations

import re
from dataclasses import dataclass

_ON_WROTE = re.compile(
    r"^On .+ wrote:\s*$|^Le .+ a écrit\s*:\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OUTLOOK_SEP = re.compile(r"^_{5,}\s*$", re.MULTILINE)
_OUTLOOK_FROM = re.compile(
    r"^(?:From|De)\s*:\s*.+$",
    re.IGNORECASE | re.MULTILINE,
)
_OUTLOOK_SENT = re.compile(
    r"^(?:Sent|Envoyé)\s*:\s*.+$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ReplyParseResult:
    new_text: str
    quoted_text: str
    confidence: float


def _split_at_first_match(text: str, patterns: list[re.Pattern[str]]) -> tuple[str, str, float]:
    earliest = len(text)
    for pattern in patterns:
        m = pattern.search(text)
        if m and m.start() < earliest:
            earliest = m.start()
    if earliest == len(text):
        return text.strip(), "", 0.5
    new_part = text[:earliest].strip()
    quoted_part = text[earliest:].strip()
    return new_part, quoted_part, 0.9


def parse_reply_body(body_text: str) -> ReplyParseResult:
    raw = (body_text or "").strip()
    if not raw:
        return ReplyParseResult(new_text="", quoted_text="", confidence=1.0)

    lines = raw.splitlines()
    cut_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(">"):
            cut_idx = idx
            break
        if _ON_WROTE.match(stripped):
            cut_idx = idx
            break
        if _OUTLOOK_SEP.match(stripped):
            cut_idx = idx
            break
        if _OUTLOOK_FROM.match(stripped) and idx + 1 < len(lines):
            next_line = lines[idx + 1].strip()
            if _OUTLOOK_SENT.match(next_line):
                cut_idx = idx
                break

    if cut_idx is not None:
        new_lines = lines[:cut_idx]
        quoted_lines = lines[cut_idx:]
        new_text = "\n".join(new_lines).strip()
        quoted_text = "\n".join(quoted_lines).strip()
        if not new_text:
            return ReplyParseResult(new_text=raw, quoted_text="", confidence=0.4)
        return ReplyParseResult(new_text=new_text, quoted_text=quoted_text, confidence=0.9)

    # Signature delimiter (-- on its own line)
    for idx, line in enumerate(lines):
        if line.strip() == "--" and idx > 0:
            new_text = "\n".join(lines[:idx]).strip()
            quoted_text = "\n".join(lines[idx:]).strip()
            if new_text:
                return ReplyParseResult(new_text=new_text, quoted_text=quoted_text, confidence=0.8)
            break

    new_text, quoted_text, confidence = _split_at_first_match(
        raw,
        [_ON_WROTE, _OUTLOOK_SEP, _OUTLOOK_FROM],
    )
    if not new_text:
        return ReplyParseResult(new_text=raw, quoted_text="", confidence=0.4)
    return ReplyParseResult(new_text=new_text, quoted_text=quoted_text, confidence=confidence)

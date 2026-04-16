from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from chatbot.adapters.rag.parsers import markdown_parser, pdf_parser, spreadsheet_parser

ParserFn = Callable[[Path], str]

_SUFFIXES: dict[str, ParserFn] = {
    ".md": markdown_parser.parse_markdown,
    ".pdf": pdf_parser.parse_pdf,
    ".csv": spreadsheet_parser.parse_csv,
    ".xlsx": spreadsheet_parser.parse_xlsx,
    ".xls": spreadsheet_parser.parse_xls,
}


def parse_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    return _SUFFIXES[suffix](path)


def supported_suffixes() -> frozenset[str]:
    return frozenset(_SUFFIXES.keys())

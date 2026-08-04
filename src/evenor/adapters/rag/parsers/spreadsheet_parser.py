from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_csv(path: Path) -> str:
    df = pd.read_csv(path)
    return _dataframe_to_text(df, sheet_name=path.name)


def parse_xlsx(path: Path) -> str:
    xls = pd.ExcelFile(path)
    parts: list[str] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        parts.append(f"## Sheet: {sheet}\n{_dataframe_to_text(df, sheet_name=sheet)}")
    return "\n\n".join(parts)


def parse_xls(path: Path) -> str:
    xls = pd.ExcelFile(path, engine="xlrd")
    parts: list[str] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, engine="xlrd")
        parts.append(f"## Sheet: {sheet}\n{_dataframe_to_text(df, sheet_name=sheet)}")
    return "\n\n".join(parts)


def _dataframe_to_text(df: pd.DataFrame, *, sheet_name: str) -> str:
    if df.empty:
        return ""
    lines: list[str] = []
    headers = [str(c) for c in df.columns]
    for _, row in df.iterrows():
        cells = [f"{h}: {row[h]}" for h in df.columns if pd.notna(row[h]) and str(row[h]).strip()]
        if cells:
            lines.append("; ".join(cells))
    return f"(columns: {', '.join(headers)})\n" + "\n".join(lines)

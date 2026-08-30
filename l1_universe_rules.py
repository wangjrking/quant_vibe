from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_ts_code(value: object) -> str | None:
    code = str(value or "").strip().upper()
    if not code or code == "NAN":
        return None
    return code


def is_bj_code(value: object) -> bool:
    code = normalize_ts_code(value)
    return bool(code and code.endswith(".BJ"))


def filter_stock_codes_no_bj(codes: Iterable[object]) -> list[str]:
    values: set[str] = set()
    for value in codes:
        code = normalize_ts_code(value)
        if code and not code.endswith(".BJ"):
            values.add(code)
    return sorted(values)


def filter_frame_no_bj(
    frame: pd.DataFrame | None,
    *,
    code_column: str = "ts_code",
) -> pd.DataFrame:
    if frame is None or frame.empty or code_column not in frame.columns:
        return pd.DataFrame() if frame is None else frame.copy()
    normalized = frame.copy()
    normalized[code_column] = normalized[code_column].astype(str).str.strip().str.upper()
    return normalized[~normalized[code_column].str.endswith(".BJ")].copy()

"""Stock pool helpers for index-constituent training and backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_INDEX_CODES = ("000300.SH", "000905.SH")
DEFAULT_INDEX_WEIGHT_DATE = "20251231"
A_SHARE_MARKETS = {"主板", "创业板", "科创板", "北交所"}


def normalize_stock_code(value) -> str | None:
    code = str(value or "").strip().upper()
    if not code or code == "NAN":
        return None
    if code.endswith(".BJ"):
        return None
    return code


def load_stock_pool(path: str | Path) -> set[str]:
    frame = pd.read_csv(path)
    if "stock_code" in frame.columns:
        series = frame["stock_code"]
    elif "con_code" in frame.columns:
        series = frame["con_code"]
    else:
        raise ValueError("stock pool file must contain stock_code or con_code")
    return {code for code in (normalize_stock_code(value) for value in series) if code}


def filter_frame_by_stock_pool(frame: pd.DataFrame, stock_pool: set[str] | None) -> pd.DataFrame:
    if not stock_pool:
        return frame
    return frame[frame["stock_code"].astype(str).str.upper().isin(stock_pool)].copy()


def normalize_a_share_code(value, include_bj: bool = True) -> str | None:
    code = str(value or "").strip().upper()
    if not code or code == "NAN":
        return None
    if not code.endswith((".SZ", ".SH", ".BJ")):
        return None
    if code.endswith(".BJ") and not include_bj:
        return None
    return code


def build_all_a_stock_pool(stock_basic: pd.DataFrame, include_bj: bool = True) -> list[str]:
    if "ts_code" not in stock_basic.columns:
        raise ValueError("stock_basic must contain ts_code")
    frame = stock_basic.copy()
    if "market" in frame.columns:
        frame = frame[frame["market"].astype(str).isin(A_SHARE_MARKETS)]
    codes = {
        code
        for code in (normalize_a_share_code(value, include_bj=include_bj) for value in frame["ts_code"])
        if code
    }
    return sorted(codes)


def fetch_index_stock_pool(ts_pro, index_codes=DEFAULT_INDEX_CODES, trade_date=DEFAULT_INDEX_WEIGHT_DATE) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for index_code in index_codes:
        weights = ts_pro.index_weight(index_code=index_code, trade_date=trade_date)
        for value in weights.get("con_code", []):
            code = normalize_stock_code(value)
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def write_stock_pool(codes: list[str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stock_code": codes}).to_csv(path, index=False, encoding="utf-8-sig")

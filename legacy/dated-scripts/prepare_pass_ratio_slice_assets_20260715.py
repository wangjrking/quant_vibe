from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
)
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
SOURCE_SIGNAL_DIR = SOURCE_REPORT_DIR / "signals" / "open_gap_deep_rebalance"
OUT_DIR = REPORT_DIR / "slice_signals"

CASES = [
    "ogd_gap1_x050",
    "ogd_gap1x70_deep5up105",
    "ogd_deep8_up110",
    "ogd_deep5_up105",
]

SLICES = [
    ("full", None),
    ("from_202407", "20240701"),
    ("from_202501", "20250102"),
    ("recent60", "__recent60__"),
]


def write_slice(case: str, slice_name: str, start: str | None) -> dict:
    source = SOURCE_SIGNAL_DIR / f"{case}.csv"
    frame = pd.read_csv(source, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep_dates = set(dates[-60:])
        sliced = frame[frame["buy_date"].astype(str).isin(keep_dates)].copy()
        effective_start = min(keep_dates) if keep_dates else ""
    elif start:
        sliced = frame[frame["buy_date"].astype(str) >= start].copy()
        effective_start = start
    else:
        sliced = frame.copy()
        effective_start = str(sliced["buy_date"].min()) if len(sliced) else ""

    out = OUT_DIR / f"{case}_{slice_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out, index=False, encoding="utf-8-sig")
    daily_counts = sliced.groupby("buy_date")["stock_code"].count() if len(sliced) else pd.Series(dtype=float)
    daily_targets = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
    return {
        "case": case,
        "slice": slice_name,
        "source_signal": str(source),
        "slice_signal": str(out),
        "start_rule": start or "full",
        "effective_start": effective_start,
        "rows": int(len(sliced)),
        "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
        "stock_count": int(sliced["stock_code"].nunique()) if len(sliced) else 0,
        "min_buy": str(sliced["buy_date"].min()) if len(sliced) else "",
        "max_buy": str(sliced["buy_date"].max()) if len(sliced) else "",
        "max_positions": int(daily_counts.max()) if len(daily_counts) else 0,
        "mean_daily_target_sum": float(daily_targets.mean()) if len(daily_targets) else 0.0,
        "max_daily_target_sum": float(daily_targets.max()) if len(daily_targets) else 0.0,
    }


def main() -> None:
    rows = [write_slice(case, slice_name, start) for case in CASES for slice_name, start in SLICES]
    manifest = pd.DataFrame(rows)
    manifest_path = REPORT_DIR / "pass_ratio_slice_signal_manifest_20260715.csv"
    json_path = REPORT_DIR / "pass_ratio_slice_signal_manifest_20260715.json"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest[["case", "slice", "rows", "buy_days", "min_buy", "max_buy", "mean_daily_target_sum"]].to_string(index=False))
    print(manifest_path)


if __name__ == "__main__":
    main()

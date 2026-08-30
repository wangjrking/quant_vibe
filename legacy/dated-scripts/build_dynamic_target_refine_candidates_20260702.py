from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
SOURCE_SIGNAL_FILE = (
    BASE_DIR
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "liq_refill_top3_p435_h1_sigpct_le_m175_bog_le1p5_s96_p10d70_fine"
    / "signals.csv"
)
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "dynamic_target_refine_candidates"


CASES = [
    {
        "name": "dyn_target_base428_boost52_cut39",
        "base": 0.428,
        "boost": 0.520,
        "cut": 0.390,
        "boost_signal_pct_max": -4.0,
        "boost_gap_pct_max": -2.0,
        "cut_signal_pct_min": -2.20,
        "cut_gap_pct_min": 0.80,
    },
    {
        "name": "dyn_target_base430_boost50_cut40",
        "base": 0.430,
        "boost": 0.500,
        "cut": 0.400,
        "boost_signal_pct_max": -4.0,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.00,
        "cut_gap_pct_min": 0.50,
    },
    {
        "name": "dyn_target_base420_tier55_45_38",
        "base": 0.420,
        "boost": 0.550,
        "mid": 0.450,
        "cut": 0.380,
        "strong_signal_pct_max": -5.0,
        "strong_gap_pct_max": 0.0,
        "mid_signal_pct_max": -3.0,
        "mid_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.20,
        "cut_gap_pct_min": 0.80,
    },
    {
        "name": "dyn_target_base432_boost49_cut41",
        "base": 0.432,
        "boost": 0.490,
        "cut": 0.410,
        "boost_signal_pct_max": -3.5,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.10,
        "cut_gap_pct_min": 0.75,
    },
    {
        "name": "dyn_target_base425_boost56_cut38",
        "base": 0.425,
        "boost": 0.560,
        "cut": 0.380,
        "boost_signal_pct_max": -4.5,
        "boost_gap_pct_max": -1.0,
        "cut_signal_pct_min": -2.30,
        "cut_gap_pct_min": 0.75,
    },
]


def _f(value) -> float:
    return float(value)


def _target(row: pd.Series, case: dict) -> float:
    signal_pct = _f(row["pct_chg"])
    gap_pct = _f(row["buy_open_gap_pct"])
    if "mid" in case:
        if signal_pct <= case["strong_signal_pct_max"] and gap_pct <= case["strong_gap_pct_max"]:
            return case["boost"]
        if signal_pct <= case["mid_signal_pct_max"] and gap_pct <= case["mid_gap_pct_max"]:
            return case["mid"]
        if signal_pct >= case["cut_signal_pct_min"] or gap_pct >= case["cut_gap_pct_min"]:
            return case["cut"]
        return case["base"]
    if signal_pct <= case["boost_signal_pct_max"] or gap_pct <= case["boost_gap_pct_max"]:
        return case["boost"]
    if signal_pct >= case["cut_signal_pct_min"] or gap_pct >= case["cut_gap_pct_min"]:
        return case["cut"]
    return case["base"]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    manifest_rows = []
    for case in CASES:
        x = df.copy()
        x["target_pct"] = x.apply(lambda row: f"{_target(row, case):.5f}", axis=1)
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        x["entry_weight_name"] = "w25_25_00_50_dynamic_target"
        x["dynamic_hold_name"] = "h1m1_dynamic_target"
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("target_pct").size().reset_index(name="rows").to_dict("records")
        manifest_rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(x)),
                "signal_days": int(x["signal_date"].nunique()),
                "target_distribution": str(counts),
                **case,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

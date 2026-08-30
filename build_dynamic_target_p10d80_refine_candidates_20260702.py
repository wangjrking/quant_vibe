from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
SOURCE_SIGNAL_FILE = (
    BASE_DIR
    / "postrank_open_filter_candidates"
    / "qfq_rerank_refill_candidates"
    / "liq_refill_top3_p383_h1_sigpct_le_m175_bog_le1p5_s96_p10d80"
    / "signals.csv"
)
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "dynamic_target_p10d80_refine_candidates"


CASES = [
    {
        "name": "p10d80_dyn_b383_boost55_sig4_gap0_cut36",
        "base": 0.383,
        "boost": 0.550,
        "cut": 0.360,
        "boost_signal_pct_max": -4.0,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.2,
        "cut_gap_pct_min": 0.8,
    },
    {
        "name": "p10d80_dyn_b390_boost52_sig4_gap0_cut36",
        "base": 0.390,
        "boost": 0.520,
        "cut": 0.360,
        "boost_signal_pct_max": -4.0,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.2,
        "cut_gap_pct_min": 0.8,
    },
    {
        "name": "p10d80_dyn_b400_boost50_sig4_gap0_cut37",
        "base": 0.400,
        "boost": 0.500,
        "cut": 0.370,
        "boost_signal_pct_max": -4.0,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.2,
        "cut_gap_pct_min": 0.8,
    },
    {
        "name": "p10d80_dyn_b390_boost58_strong_cut35",
        "base": 0.390,
        "boost": 0.580,
        "mid": 0.440,
        "cut": 0.350,
        "strong_signal_pct_max": -5.0,
        "strong_gap_pct_max": 0.0,
        "mid_signal_pct_max": -3.0,
        "mid_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.2,
        "cut_gap_pct_min": 0.8,
    },
    {
        "name": "p10d80_dyn_b405_boost49_sig35_cut38",
        "base": 0.405,
        "boost": 0.490,
        "cut": 0.380,
        "boost_signal_pct_max": -3.5,
        "boost_gap_pct_max": 0.0,
        "cut_signal_pct_min": -2.1,
        "cut_gap_pct_min": 0.8,
    },
]


def _target(row: pd.Series, case: dict) -> float:
    signal_pct = float(row["pct_chg"])
    gap_pct = float(row["buy_open_gap_pct"])
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
    rows = []
    for case in CASES:
        x = df.copy()
        x["target_pct"] = x.apply(lambda row: f"{_target(row, case):.5f}", axis=1)
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        x["entry_weight_name"] = "w25_25_00_50_p10d80_dynamic_target"
        x["dynamic_hold_name"] = "h1m1_p10d80_dynamic_target"
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")
        rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(x)),
                "signal_days": int(x["signal_date"].nunique()),
                "target_distribution": str(x.groupby("target_pct").size().reset_index(name="rows").to_dict("records")),
                **case,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

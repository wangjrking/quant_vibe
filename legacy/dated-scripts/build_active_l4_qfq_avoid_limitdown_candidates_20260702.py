from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
BASE_DIR = REPORT_DIR / "active_l4_qfq_rebuild"
SOURCE_SIGNAL_FILE = BASE_DIR / "active_l4_qfq_best_rebuild_p435" / "signals.csv"
OUT_DIR = BASE_DIR / "avoid_limitdown_candidates"


CASES = [
    {
        "name": "qfq_no_limitdown_top3_p35_pct_m8_to_m175",
        "pct_min": -8.0,
        "pct_max": -1.75,
        "gap_high": 0.015,
        "p10d_min": 0.70,
        "topn": 3,
        "target_pct": "0.35000",
    },
    {
        "name": "qfq_no_limitdown_top3_p435_pct_m8_to_m175",
        "pct_min": -8.0,
        "pct_max": -1.75,
        "gap_high": 0.015,
        "p10d_min": 0.70,
        "topn": 3,
        "target_pct": "0.43500",
    },
    {
        "name": "qfq_midpull_top3_p435_pct_m6_to_m175",
        "pct_min": -6.0,
        "pct_max": -1.75,
        "gap_high": 0.015,
        "p10d_min": 0.70,
        "topn": 3,
        "target_pct": "0.43500",
    },
    {
        "name": "qfq_midpull_top2_p50_pct_m6_to_m175",
        "pct_min": -6.0,
        "pct_max": -1.75,
        "gap_high": 0.015,
        "p10d_min": 0.70,
        "topn": 2,
        "target_pct": "0.50000",
    },
    {
        "name": "qfq_softpull_top3_p435_pct_m4_to_m075",
        "pct_min": -4.0,
        "pct_max": -0.75,
        "gap_high": 0.015,
        "p10d_min": 0.70,
        "topn": 3,
        "target_pct": "0.43500",
    },
    {
        "name": "qfq_no_limitdown_top3_p435_pct_m8_to_m175_p10d90",
        "pct_min": -8.0,
        "pct_max": -1.75,
        "gap_high": 0.015,
        "p10d_min": 0.90,
        "topn": 3,
        "target_pct": "0.43500",
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows = []
    for case in CASES:
        x = df[
            (df["pct_chg"] >= float(case["pct_min"]))
            & (df["pct_chg"] <= float(case["pct_max"]))
            & (df["buy_open_gap_pct"] <= float(case["gap_high"]) * 100.0)
            & (df["pred_10d"] >= float(case["p10d_min"]))
        ].copy()
        if x.empty:
            continue
        x = x.sort_values(["signal_date", "entry_score", "stock_code"], ascending=[True, False, True])
        x = x.groupby("signal_date", group_keys=False).head(int(case["topn"]))
        x["rank"] = x.groupby("signal_date").cumcount() + 1
        x["target_pct"] = case["target_pct"]
        x["holding_days"] = 1
        x["max_holding_days"] = 1
        x["score_exit_entry_ratio"] = "9.99000"
        x["min_holding_days_before_score_exit"] = 1
        x["score_continue_entry_ratio"] = "9.99000"
        x["strategy_variant"] = case["name"]
        x["filter_name"] = case["name"]
        x["dynamic_hold_name"] = "h1m1_qfq_avoid_limitdown"
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        x.to_csv(signal_file, index=False, encoding="utf-8")
        counts = x.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "score_db": str(BASE_DIR / "score.duckdb"),
                "score_table": "score",
                "rows": int(len(x)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                "days_below_topn": int((counts < int(case["topn"])).sum()),
                **case,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8-sig")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

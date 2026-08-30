from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
BASE_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_latest_l4_weight_candidates_20260702"
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "score_exit_rule_candidates"
SOURCE_SIGNAL_FILE = BASE_DIR / "postrank_open_filter_candidates" / "low_to_flat_pct12_top3_pos40" / "signals.csv"


CASES = [
    {
        "name": "exitratio97_h2m2_p40",
        "target_pct": "0.40000",
        "holding_days": 2,
        "max_holding_days": 2,
        "score_exit_entry_ratio": "0.97000",
        "score_continue_entry_ratio": "9.99000",
        "score_exit_rank": None,
    },
    {
        "name": "exitratio95_h2m2_p40",
        "target_pct": "0.40000",
        "holding_days": 2,
        "max_holding_days": 2,
        "score_exit_entry_ratio": "0.95000",
        "score_continue_entry_ratio": "9.99000",
        "score_exit_rank": None,
    },
    {
        "name": "exitrank70_h2m3_p40",
        "target_pct": "0.40000",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_entry_ratio": "0.95000",
        "score_continue_entry_ratio": "0.98000",
        "score_exit_rank": 0.70,
    },
    {
        "name": "exitrank80_h2m3_p40",
        "target_pct": "0.40000",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_entry_ratio": "0.95000",
        "score_continue_entry_ratio": "0.98000",
        "score_exit_rank": 0.80,
    },
    {
        "name": "exitrank70_h2m3_p34",
        "target_pct": "0.34000",
        "holding_days": 2,
        "max_holding_days": 3,
        "score_exit_entry_ratio": "0.95000",
        "score_continue_entry_ratio": "0.98000",
        "score_exit_rank": 0.70,
    },
    {
        "name": "exitrank80_h3m3_p34",
        "target_pct": "0.34000",
        "holding_days": 3,
        "max_holding_days": 3,
        "score_exit_entry_ratio": "0.95000",
        "score_continue_entry_ratio": "9.99000",
        "score_exit_rank": 0.80,
    },
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(SOURCE_SIGNAL_FILE, dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows = []
    for case in CASES:
        df = src.copy()
        df["target_pct"] = case["target_pct"]
        df["holding_days"] = int(case["holding_days"])
        df["max_holding_days"] = int(case["max_holding_days"])
        df["score_exit_entry_ratio"] = case["score_exit_entry_ratio"]
        df["min_holding_days_before_score_exit"] = 1
        df["score_continue_entry_ratio"] = case["score_continue_entry_ratio"]
        df["strategy_variant"] = case["name"]
        out_dir = OUT_DIR / case["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        signal_file = out_dir / "signals.csv"
        df.to_csv(signal_file, index=False, encoding="utf-8")
        counts = df.groupby("signal_date").size()
        rows.append(
            {
                "name": case["name"],
                "signal_file": str(signal_file),
                "rows": int(len(df)),
                "signal_days": int(counts.size),
                "avg_names": float(counts.mean()),
                **case,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT_DIR / "candidate_manifest.csv", index=False, encoding="utf-8")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

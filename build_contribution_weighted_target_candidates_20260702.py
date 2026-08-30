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
OUT_DIR = BASE_DIR / "postrank_open_filter_candidates" / "contribution_weighted_target_candidates"


CASES = [
    {
        "name": "cwt_strong55_weak28_base42",
        "base": 0.42,
        "strong": 0.55,
        "weak": 0.28,
        "strong_pct_max": -4.0,
        "strong_gap_max": -0.38,
        "weak_pct_min": -2.90,
        "weak_gap_min": 0.25,
        "weak_pred1d_min": 0.9985,
    },
    {
        "name": "cwt_strong60_weak25_base40",
        "base": 0.40,
        "strong": 0.60,
        "weak": 0.25,
        "strong_pct_max": -4.0,
        "strong_gap_max": -0.38,
        "weak_pct_min": -2.90,
        "weak_gap_min": 0.25,
        "weak_pred1d_min": 0.9985,
    },
    {
        "name": "cwt_strong65_weak20_base38",
        "base": 0.38,
        "strong": 0.65,
        "weak": 0.20,
        "strong_pct_max": -4.0,
        "strong_gap_max": -0.38,
        "weak_pct_min": -2.90,
        "weak_gap_min": 0.25,
        "weak_pred1d_min": 0.9985,
    },
    {
        "name": "cwt_strong58_weak30_base43",
        "base": 0.43,
        "strong": 0.58,
        "weak": 0.30,
        "strong_pct_max": -3.50,
        "strong_gap_max": -0.20,
        "weak_pct_min": -2.70,
        "weak_gap_min": 0.00,
        "weak_pred1d_min": 0.9990,
    },
    {
        "name": "cwt_strong62_weak22_base39",
        "base": 0.39,
        "strong": 0.62,
        "weak": 0.22,
        "strong_pct_max": -4.50,
        "strong_gap_max": 0.00,
        "weak_pct_min": -2.70,
        "weak_gap_min": 0.25,
        "weak_pred1d_min": 0.9990,
    },
]


def _target(row: pd.Series, case: dict) -> float:
    pct = float(row["pct_chg"])
    gap = float(row["buy_open_gap_pct"])
    pred1d = float(row["pred_1d"])
    if (
        pct >= case["weak_pct_min"]
        or gap >= case["weak_gap_min"]
        or pred1d >= case["weak_pred1d_min"]
    ):
        return case["weak"]
    if pct <= case["strong_pct_max"] and gap <= case["strong_gap_max"]:
        return case["strong"]
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
        x["entry_weight_name"] = "w25_25_00_50_contribution_weighted"
        x["dynamic_hold_name"] = "h1m1_contribution_weighted"
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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_open_quality_pick1"
LOG_DIR = REPORT_DIR / "logs" / "top3_open_quality_pick1"
BASE_SIGNAL = (
    ROOT
    / "quant"
    / "main"
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "signals"
    / "full_history_fw_soft_deepdrop_weight.csv"
)


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


def load_all() -> pd.DataFrame:
    df = pd.read_csv(BASE_SIGNAL, encoding="utf-8-sig")
    for col in [
        "rank",
        "target_pct",
        "pred_prob",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_pct",
        "buy_open_gap_raw_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["exec_open_gap_pct"] = df["buy_open_gap_raw_pct"].fillna(df["buy_open_gap_pct"])
    return df


def write_pick_signal(base: pd.DataFrame, case: dict) -> dict:
    df = base.copy()
    high, mid = base_mod.quality_flags(df, case["profile"])
    df["quality_bucket"] = 0
    df.loc[mid, "quality_bucket"] = 1
    df.loc[high, "quality_bucket"] = 2
    df["quality_target"] = case["low"]
    df.loc[mid, "quality_target"] = case["mid"]
    df.loc[high, "quality_target"] = case["high"]
    if case["sort_by"] == "quality_pred10":
        sort_cols = ["buy_date", "quality_bucket", "pred_10d", "pred_prob", "rank"]
        asc = [True, False, False, False, True]
    elif case["sort_by"] == "quality_gap":
        # Prefer higher quality, then smaller absolute open gap, then model score.
        df["abs_gap"] = df["exec_open_gap_pct"].abs()
        sort_cols = ["buy_date", "quality_bucket", "abs_gap", "pred_10d", "rank"]
        asc = [True, False, True, False, True]
    else:
        sort_cols = ["buy_date", "quality_bucket", "rank", "pred_prob"]
        asc = [True, False, True, False]
    selected = df.sort_values(sort_cols, ascending=asc).groupby("buy_date", as_index=False).head(1).copy()
    selected["target_pct"] = selected["quality_target"]
    selected = selected[selected["target_pct"] > 0].copy()
    selected["strategy_variant"] = case["case"]
    selected["filter_name"] = case["case"]
    selected["daily_target_sum_after_cap"] = selected["target_pct"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "profile": case["profile"],
        "sort_by": case["sort_by"],
        "high": case["high"],
        "mid": case["mid"],
        "low": case["low"],
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()),
        "high_rows": int((selected["quality_bucket"] == 2).sum()),
        "mid_rows": int((selected["quality_bucket"] == 1).sum()),
        "low_rows": int((selected["quality_bucket"] == 0).sum()),
        "mean_target": float(selected["target_pct"].mean()) if len(selected) else 0.0,
        "rank1_rows": int((selected["rank"] == 1).sum()),
        "rank2_rows": int((selected["rank"] == 2).sum()),
        "rank3_rows": int((selected["rank"] == 3).sum()),
    }


def main() -> None:
    base = load_all()
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    cases = []
    profiles = ["mid_atr12", "mid_gap_amt", "mid_deep_only"]
    sorts = ["quality_pred10", "quality_gap"]
    weights = [
        (0.80, 0.35, 0.15),
        (0.78, 0.36, 0.18),
        (0.76, 0.34, 0.22),
        (0.74, 0.34, 0.24),
        (0.72, 0.34, 0.26),
        (0.85, 0.35, 0.10),
        (0.91, 0.35, 0.05),
    ]
    for profile in profiles:
        for sort_by in sorts:
            for high, mid, low in weights:
                cases.append(
                    {
                        "case": f"pick1_{profile}_{sort_by}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}",
                        "profile": profile,
                        "sort_by": sort_by,
                        "high": high,
                        "mid": mid,
                        "low": low,
                    }
                )
    manifest = [write_pick_signal(base, case) for case in cases]
    results = [base_mod.run_juejin(row) for row in manifest]
    csv_path = REPORT_DIR / "top3_open_quality_pick1_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_open_quality_pick1_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

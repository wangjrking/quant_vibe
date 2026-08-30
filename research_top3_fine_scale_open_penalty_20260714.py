from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "quant" / "main" / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_fine_scale_open_penalty"
LOG_DIR = REPORT_DIR / "logs" / "top3_fine_scale_open_penalty"
SOURCE = REPORT_DIR / "signals" / "top3_multi_open_quality_sizing" / "mf_t2_sc1200_cap100.csv"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = []
for scale in [0.930, 0.940, 0.950, 0.955, 0.960, 0.965]:
    CASES.append({"scale": scale, "pos_gap_cut": None, "pos_gap_mult": 1.0, "shallow_cut": None, "shallow_mult": 1.0})
for scale in [0.955, 0.960, 0.965, 0.970]:
    for pos_gap_cut, pos_gap_mult in [(0.0, 0.90), (0.5, 0.85), (1.0, 0.80)]:
        CASES.append(
            {
                "scale": scale,
                "pos_gap_cut": pos_gap_cut,
                "pos_gap_mult": pos_gap_mult,
                "shallow_cut": None,
                "shallow_mult": 1.0,
            }
        )
for scale in [0.965, 0.970, 0.975]:
    for shallow_cut, shallow_mult in [(-2.0, 0.92), (-1.75, 0.90), (-1.5, 0.88)]:
        CASES.append(
            {
                "scale": scale,
                "pos_gap_cut": 0.5,
                "pos_gap_mult": 0.88,
                "shallow_cut": shallow_cut,
                "shallow_mult": shallow_mult,
            }
        )


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(index=df.index, dtype=float)


def write_case(base: pd.DataFrame, case: dict) -> dict:
    df = base.copy()
    target = numeric(df, "target_pct").fillna(0.0) * case["scale"]
    gap = numeric(df, "exec_open_gap_pct")
    if gap.isna().all():
        gap = numeric(df, "buy_open_gap_raw_pct").fillna(numeric(df, "buy_open_gap_pct"))
    pct = numeric(df, "signal_pct_chg_raw")
    if case["pos_gap_cut"] is not None:
        target = target.where(~(gap > case["pos_gap_cut"]), target * case["pos_gap_mult"])
    if case["shallow_cut"] is not None:
        target = target.where(~(pct > case["shallow_cut"]), target * case["shallow_mult"])
    daily_sum = target.groupby(df["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = target * cap_scale
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = "0.98000"
    df["score_continue_entry_ratio"] = "1.02000"
    df["min_holding_days_before_score_exit"] = 1
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case_name = (
        f"fsp_sc{int(case['scale'] * 1000):03d}"
        f"_pg{case['pos_gap_cut'] if case['pos_gap_cut'] is not None else 'none'}"
        f"m{int(case['pos_gap_mult'] * 100):03d}"
        f"_sh{case['shallow_cut'] if case['shallow_cut'] is not None else 'none'}"
        f"m{int(case['shallow_mult'] * 100):03d}"
    ).replace(".", "p").replace("-", "m")
    df["strategy_variant"] = case_name
    df["filter_name"] = case_name
    out = SIGNAL_DIR / f"{case_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "mean_target": float(df["target_pct"].mean()) if len(df) else 0.0,
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()) if len(df) else 0.0,
        **case,
    }


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(SOURCE, encoding="utf-8-sig")
    if "exec_open_gap_pct" not in base.columns:
        base["exec_open_gap_pct"] = numeric(base, "buy_open_gap_raw_pct").fillna(numeric(base, "buy_open_gap_pct"))
    manifest = [write_case(base, case) for case in CASES]
    results = []
    for row in manifest:
        result = base_mod.run_juejin(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "pnl_ratio_annual": result.get("pnl_ratio_annual"),
                    "sharp_ratio": result.get("sharp_ratio"),
                    "max_drawdown": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    csv_path = REPORT_DIR / "top3_fine_scale_open_penalty_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_fine_scale_open_penalty_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

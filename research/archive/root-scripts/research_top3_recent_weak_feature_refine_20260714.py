from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MULTI_SCRIPT = ROOT / "quant" / "main" / "research_top3_multi_open_quality_sizing_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_recent_weak_feature_refine"
LOG_DIR = REPORT_DIR / "logs" / "top3_recent_weak_feature_refine"


spec = importlib.util.spec_from_file_location("multi", MULTI_SCRIPT)
multi_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(multi_mod)


def _num(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return float(value)


def row_target(row: pd.Series, case: dict) -> float:
    bucket = int(row.get("quality_bucket", 0))
    pct = _num(row, "signal_pct_chg_raw")
    gap = _num(row, "exec_open_gap_pct", 99.0)
    amount = _num(row, "amount")
    pred1 = _num(row, "pred_1d")
    atr = _num(row, "atr_qfq")

    target = case["high"] if bucket == 2 else case["mid"] if bucket == 1 else case["low"]

    if pct > -3.0:
        target *= case["shallow_scale"]
    elif pct <= -5.0:
        target *= case["deep_scale"]

    if gap > 0.5:
        target *= case["pos_gap_scale"]
    elif gap <= -3.0:
        target *= case["deep_gap_scale"]

    if 0.50 <= pred1 < 0.90 and bucket < 2:
        target *= case["mid_pred1_scale"]
    if amount < case["low_amount_cut"] and bucket == 0:
        target *= case["low_liq_scale"]
    if atr >= case["high_atr_cut"] and bucket == 0:
        target *= case["low_atr_scale"]

    return max(0.0, target)


def write_case(base: pd.DataFrame, case: dict) -> dict:
    df = base.copy()
    high_mask, mid_mask = multi_mod.pick_mod.base_mod.quality_flags(df, "mid_atr12")
    df["quality_bucket"] = 0
    df.loc[mid_mask, "quality_bucket"] = 1
    df.loc[high_mask, "quality_bucket"] = 2

    df["sort_score"] = (
        df["quality_bucket"].astype(float) * 10.0
        + pd.to_numeric(df["pred_10d"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["pred_1d"], errors="coerce").fillna(0.0) * case["pred1_sort_weight"]
        + (pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).clip(upper=500000.0) / 500000.0)
        * case["amount_sort_weight"]
    )
    selected = (
        df.sort_values(["buy_date", "sort_score", "rank"], ascending=[True, False, True])
        .groupby("buy_date", group_keys=False)
        .head(case["top_n"])
        .copy()
    )
    selected["target_pct"] = selected.apply(lambda row: row_target(row, case), axis=1)
    selected = selected[selected["target_pct"] > 0].copy()
    sums = selected.groupby("buy_date")["target_pct"].transform("sum")
    scale = (case["daily_cap"] / sums).clip(upper=1.0)
    selected["target_pct"] = selected["target_pct"] * scale
    selected["strategy_variant"] = case["case"]
    selected["filter_name"] = case["case"]
    selected["daily_target_sum_after_cap"] = selected.groupby("buy_date")["target_pct"].transform("sum")
    selected["rank"] = selected.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)

    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(selected)),
        "buy_days": int(selected["buy_date"].nunique()) if len(selected) else 0,
        "stock_count": int(selected["stock_code"].nunique()) if len(selected) else 0,
        "mean_target": float(selected["target_pct"].mean()) if len(selected) else 0.0,
        "mean_daily_target_sum": float(selected.groupby("buy_date")["target_pct"].sum().mean()) if len(selected) else 0.0,
        **{k: v for k, v in case.items() if k != "case"},
    }


def build_cases() -> list[dict]:
    cases: list[dict] = []
    for top_n in [2, 3]:
        for high, mid, low in [(0.72, 0.36, 0.14), (0.78, 0.39, 0.14), (0.84, 0.42, 0.12)]:
            for mid_pred1_scale in [0.45, 0.70]:
                for amount_sort_weight in [0.00, 0.08]:
                    cases.append(
                        {
                            "case": (
                                f"rwf_t{top_n}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}"
                                f"_p1{int(mid_pred1_scale*100):02d}_liq{int(amount_sort_weight*100):02d}"
                            ),
                            "top_n": top_n,
                            "high": high,
                            "mid": mid,
                            "low": low,
                            "shallow_scale": 0.75,
                            "deep_scale": 1.00,
                            "pos_gap_scale": 0.50,
                            "deep_gap_scale": 1.05,
                            "daily_cap": 1.00,
                            "mid_pred1_scale": mid_pred1_scale,
                            "low_amount_cut": 200000.0,
                            "low_liq_scale": 0.55,
                            "high_atr_cut": 8.0,
                            "low_atr_scale": 0.70,
                            "pred1_sort_weight": 0.05,
                            "amount_sort_weight": amount_sort_weight,
                        }
                    )
    return cases


def main() -> None:
    base = multi_mod.pick_mod.load_all()
    multi_mod.pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR
    multi_mod.pick_mod.base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = [write_case(base, case) for case in build_cases()]
    results = []
    for row in manifest:
        result = multi_mod.pick_mod.base_mod.run_juejin(row)
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    csv_path = REPORT_DIR / "top3_recent_weak_feature_refine_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_recent_weak_feature_refine_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

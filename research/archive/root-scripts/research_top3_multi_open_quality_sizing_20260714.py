from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PICK_SCRIPT = ROOT / "quant" / "main" / "research_top3_open_quality_pick1_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_multi_open_quality_sizing"
LOG_DIR = REPORT_DIR / "logs" / "top3_multi_open_quality_sizing"


spec = importlib.util.spec_from_file_location("pick1", PICK_SCRIPT)
pick_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pick_mod)


def row_target(row: pd.Series, case: dict) -> float:
    bucket = int(row.get("quality_bucket", 0))
    pct = float(row.get("signal_pct_chg_raw", 0))
    gap = row.get("exec_open_gap_pct")
    gap = float(gap) if pd.notna(gap) else 99.0
    target = case["high"] if bucket == 2 else case["mid"] if bucket == 1 else case["low"]
    if pct > -3.0:
        target *= case["shallow_scale"]
    elif pct <= -5.0:
        target *= case["deep_scale"]
    if gap > 0.5:
        target *= case["pos_gap_scale"]
    elif gap <= -3.0:
        target *= case["deep_gap_scale"]
    return max(0.0, target)


def write_case(base: pd.DataFrame, case: dict) -> dict:
    df = base.copy()
    high_mask, mid_mask = pick_mod.base_mod.quality_flags(df, "mid_atr12")
    df["quality_bucket"] = 0
    df.loc[mid_mask, "quality_bucket"] = 1
    df.loc[high_mask, "quality_bucket"] = 2
    df["sort_score"] = (
        df["quality_bucket"].astype(float) * 10.0
        + pd.to_numeric(df["pred_10d"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["pred_1d"], errors="coerce").fillna(0.0) * 0.05
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
        "top_n": case["top_n"],
        "high": case["high"],
        "mid": case["mid"],
        "low": case["low"],
        "shallow_scale": case["shallow_scale"],
        "pos_gap_scale": case["pos_gap_scale"],
        "deep_scale": case["deep_scale"],
        "deep_gap_scale": case["deep_gap_scale"],
        "daily_cap": case["daily_cap"],
    }


def main() -> None:
    base = pick_mod.load_all()
    pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR
    pick_mod.base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for top_n in [2, 3]:
        for high, mid, low in [(0.60, 0.30, 0.18), (0.70, 0.34, 0.20), (0.80, 0.36, 0.18)]:
            for shallow_scale in [0.50, 0.75]:
                for pos_gap_scale in [0.35, 0.50]:
                    cases.append(
                        {
                            "case": (
                                f"multi_t{top_n}_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}"
                                f"_ss{int(shallow_scale*100):02d}_pg{int(pos_gap_scale*100):02d}"
                            ),
                            "top_n": top_n,
                            "high": high,
                            "mid": mid,
                            "low": low,
                            "shallow_scale": shallow_scale,
                            "pos_gap_scale": pos_gap_scale,
                            "deep_scale": 1.00,
                            "deep_gap_scale": 1.05,
                            "daily_cap": 0.91,
                        }
                    )

    manifest = [write_case(base, case) for case in cases]
    results = []
    for row in manifest:
        result = pick_mod.base_mod.run_juejin(row)
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

    csv_path = REPORT_DIR / "top3_multi_open_quality_sizing_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_multi_open_quality_sizing_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

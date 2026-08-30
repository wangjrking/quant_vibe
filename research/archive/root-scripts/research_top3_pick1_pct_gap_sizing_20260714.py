from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PICK_SCRIPT = ROOT / "quant" / "main" / "research_top3_open_quality_pick1_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_pick1_pct_gap_sizing"
LOG_DIR = REPORT_DIR / "logs" / "top3_pick1_pct_gap_sizing"


spec = importlib.util.spec_from_file_location("pick1", PICK_SCRIPT)
pick_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(pick_mod)


def target_for_row(row: pd.Series, case: dict) -> float:
    bucket = int(row.get("quality_bucket", 0))
    pct = float(row.get("signal_pct_chg_raw", 0))
    gap = row.get("exec_open_gap_pct")
    gap = float(gap) if pd.notna(gap) else 99.0
    if bucket == 2:
        target = case["high"]
    elif bucket == 1:
        target = case["mid"]
    else:
        target = case["low"]

    if pct > -3.0:
        target *= case["shallow_scale"]
    elif pct <= -5.0:
        target *= case["deep_scale"]
    if gap > 0.5:
        target *= case["pos_gap_scale"]
    elif gap <= -3.0:
        target *= case["deep_gap_scale"]
    return max(0.0, min(case["cap"], target))


def write_case(base: pd.DataFrame, case: dict) -> dict:
    selected_meta = pick_mod.write_pick_signal(
        base,
        {
            "case": case["case"] + "_select_tmp",
            "profile": "mid_atr12",
            "sort_by": "quality_pred10",
            "high": case["high"],
            "mid": case["mid"],
            "low": case["low"],
        },
    )
    df = pd.read_csv(selected_meta["signal_file"], encoding="utf-8-sig")
    for col in [
        "target_pct",
        "quality_bucket",
        "signal_pct_chg_raw",
        "exec_open_gap_pct",
        "pred_10d",
        "rank",
        "atr_qfq",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target_pct"] = df.apply(lambda row: target_for_row(row, case), axis=1)
    df = df[df["target_pct"] > 0].copy()
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["daily_target_sum_after_cap"] = df["target_pct"]
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "mean_target": float(df["target_pct"].mean()) if len(df) else 0.0,
        "high": case["high"],
        "mid": case["mid"],
        "low": case["low"],
        "shallow_scale": case["shallow_scale"],
        "deep_scale": case["deep_scale"],
        "pos_gap_scale": case["pos_gap_scale"],
        "deep_gap_scale": case["deep_gap_scale"],
        "cap": case["cap"],
    }


def main() -> None:
    base = pick_mod.load_all()
    pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR / "_tmp"
    pick_mod.base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    base_weights = [
        (0.68, 0.36, 0.28),
        (0.70, 0.36, 0.26),
        (0.72, 0.36, 0.24),
        (0.66, 0.36, 0.30),
    ]
    shallow_scales = [0.45, 0.55, 0.65, 0.75]
    pos_gap_scales = [0.35, 0.50, 0.65]
    deep_scales = [1.00, 1.08]
    for high, mid, low in base_weights:
        for shallow in shallow_scales:
            for pos_gap in pos_gap_scales:
                for deep in deep_scales:
                    cases.append(
                        {
                            "case": (
                                f"pg_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}"
                                f"_ss{int(shallow*100):02d}_pg{int(pos_gap*100):02d}_ds{int(deep*100):03d}"
                            ),
                            "high": high,
                            "mid": mid,
                            "low": low,
                            "shallow_scale": shallow,
                            "pos_gap_scale": pos_gap,
                            "deep_scale": deep,
                            "deep_gap_scale": 1.0,
                            "cap": 0.91,
                        }
                    )
    manifest = [write_case(base, case) for case in cases]
    results = [pick_mod.base_mod.run_juejin(row) for row in manifest]
    csv_path = REPORT_DIR / "top3_pick1_pct_gap_sizing_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_pick1_pct_gap_sizing_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

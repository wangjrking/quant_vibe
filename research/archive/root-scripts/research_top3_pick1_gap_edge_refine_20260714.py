from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PICK_SCRIPT = ROOT / "quant" / "main" / "research_top3_open_quality_pick1_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_pick1_gap_edge_refine"
LOG_DIR = REPORT_DIR / "logs" / "top3_pick1_gap_edge_refine"


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
    pick_mod.SIGNAL_DIR = SIGNAL_DIR / "_pick_tmp"
    pick_mod.LOG_DIR = LOG_DIR
    pick_mod.base_mod.SIGNAL_DIR = SIGNAL_DIR / "_pick_tmp"
    pick_mod.base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    base_weights = [
        (0.66, 0.36, 0.30),
        (0.67, 0.36, 0.29),
        (0.68, 0.36, 0.28),
        (0.69, 0.36, 0.27),
        (0.70, 0.36, 0.26),
    ]
    for high, mid, low in base_weights:
        for shallow in [0.75, 0.80]:
            for pos_gap in [0.20, 0.35]:
                for deep_gap in [1.05, 1.10]:
                    cases.append(
                        {
                            "case": (
                                f"ge_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}"
                                f"_ss{int(shallow*100):02d}_pg{int(pos_gap*100):02d}_ds100"
                                f"_dg{int(deep_gap*100):03d}"
                            ),
                            "high": high,
                            "mid": mid,
                            "low": low,
                            "shallow_scale": shallow,
                            "pos_gap_scale": pos_gap,
                            "deep_scale": 1.00,
                            "deep_gap_scale": deep_gap,
                            "cap": 0.91,
                        }
                    )
    for high, mid, low in [(0.70, 0.36, 0.26), (0.72, 0.36, 0.24)]:
        for pos_gap in [0.20, 0.35]:
            for deep in [1.02, 1.04]:
                cases.append(
                    {
                        "case": (
                            f"ge_h{int(high*100):02d}_m{int(mid*100):02d}_l{int(low*100):02d}"
                            f"_ss75_pg{int(pos_gap*100):02d}_ds{int(deep*100):03d}_dg105"
                        ),
                        "high": high,
                        "mid": mid,
                        "low": low,
                        "shallow_scale": 0.75,
                        "pos_gap_scale": pos_gap,
                        "deep_scale": deep,
                        "deep_gap_scale": 1.05,
                        "cap": 0.91,
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
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    csv_path = REPORT_DIR / "top3_pick1_gap_edge_refine_juejin_results_20260714.csv"
    json_path = REPORT_DIR / "top3_pick1_gap_edge_refine_juejin_results_20260714.json"
    pd.DataFrame(results).sort_values(
        ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
    ).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

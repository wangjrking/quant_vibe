from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SOURCE = REPORT_DIR / "signals" / "highsharpe_position_scale" / "hps_sc092_x1040_cap100.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "open_gap_soft_gate_refine"
LOG_DIR = REPORT_DIR / "logs" / "open_gap_soft_gate_refine"
OUT_CSV = REPORT_DIR / "open_gap_soft_gate_refine_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "open_gap_soft_gate_refine_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "ogs_posgt0_x080", "pos_gt": 0.0, "pos_scale": 0.80, "neg_lt": None, "neg_scale": 1.00, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.05},
    {"case": "ogs_posgt0_x065", "pos_gt": 0.0, "pos_scale": 0.65, "neg_lt": None, "neg_scale": 1.00, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.08},
    {"case": "ogs_posgt05_x070", "pos_gt": 0.5, "pos_scale": 0.70, "neg_lt": None, "neg_scale": 1.00, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.06},
    {"case": "ogs_posgt10_x050", "pos_gt": 1.0, "pos_scale": 0.50, "neg_lt": None, "neg_scale": 1.00, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.05},
    {"case": "ogs_neglt5_x080", "pos_gt": None, "pos_scale": 1.00, "neg_lt": -5.0, "neg_scale": 0.80, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.06},
    {"case": "ogs_neglt6_x065", "pos_gt": None, "pos_scale": 1.00, "neg_lt": -6.0, "neg_scale": 0.65, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.08},
    {"case": "ogs_pos0_neg6", "pos_gt": 0.0, "pos_scale": 0.75, "neg_lt": -6.0, "neg_scale": 0.75, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.08},
    {"case": "ogs_pos05_neg6", "pos_gt": 0.5, "pos_scale": 0.70, "neg_lt": -6.0, "neg_scale": 0.75, "deep_pct": None, "deep_scale": 1.00, "global_scale": 1.08},
    {"case": "ogs_deeppct5_x090", "pos_gt": None, "pos_scale": 1.00, "neg_lt": None, "neg_scale": 1.00, "deep_pct": -5.0, "deep_scale": 0.90, "global_scale": 1.06},
    {"case": "ogs_deeppct6_x075", "pos_gt": None, "pos_scale": 1.00, "neg_lt": None, "neg_scale": 1.00, "deep_pct": -6.0, "deep_scale": 0.75, "global_scale": 1.08},
    {"case": "ogs_pos0_deep5", "pos_gt": 0.0, "pos_scale": 0.75, "neg_lt": None, "neg_scale": 1.00, "deep_pct": -5.0, "deep_scale": 0.85, "global_scale": 1.10},
    {"case": "ogs_pos05_neg6_deep5", "pos_gt": 0.5, "pos_scale": 0.75, "neg_lt": -6.0, "neg_scale": 0.80, "deep_pct": -5.0, "deep_scale": 0.90, "global_scale": 1.10},
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def write_variant(source: pd.DataFrame, case: dict) -> dict:
    df = source.copy()
    target = _num(df["target_pct"]).fillna(0.0)
    scale = pd.Series(float(case["global_scale"]), index=df.index)
    gap = _num(df.get("exec_open_gap_pct", df.get("buy_open_gap_raw_pct", df.get("buy_open_gap_pct"))))
    pct = _num(df.get("signal_pct_chg_raw"))

    if case["pos_gt"] is not None:
        scale = scale.mask(gap > float(case["pos_gt"]), scale * float(case["pos_scale"]))
    if case["neg_lt"] is not None:
        scale = scale.mask(gap < float(case["neg_lt"]), scale * float(case["neg_scale"]))
    if case["deep_pct"] is not None:
        scale = scale.mask(pct < float(case["deep_pct"]), scale * float(case["deep_scale"]))

    df["target_pct"] = (target * scale).clip(upper=1.0)
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["open_gap_soft_gate_case"] = case["case"]

    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        **case,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "pos_scaled_rows": int((gap > float(case["pos_gt"])).sum()) if case["pos_gt"] is not None else 0,
        "neg_scaled_rows": int((gap < float(case["neg_lt"])).sum()) if case["neg_lt"] is not None else 0,
        "deep_scaled_rows": int((pct < float(case["deep_pct"])).sum()) if case["deep_pct"] is not None else 0,
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    manifest = [write_variant(source, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
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
    frame = pd.DataFrame(results)
    frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    print(
        json.dumps(
            {"csv": str(OUT_CSV), "json": str(OUT_JSON), "cases": len(results), "target_hits": int(len(hits))},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

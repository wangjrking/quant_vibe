from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CORE = REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_gap1x50_deep8up110.csv"
OVERLAY = REPORT_DIR / "signals" / "pool_pullback_gap_rule" / "pgr_w10p1_t1_p990_pct8_15_gap5_0_pos20.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_pullback_overlay"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_pullback_overlay"
OUT_CSV = REPORT_DIR / "core_plus_pullback_overlay_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_plus_pullback_overlay_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {"case": "cpo_empty_o06_cap100", "mode": "empty_only", "overlay_target": 0.06, "cap": 1.0},
    {"case": "cpo_empty_o10_cap100", "mode": "empty_only", "overlay_target": 0.10, "cap": 1.0},
    {"case": "cpo_empty_o15_cap100", "mode": "empty_only", "overlay_target": 0.15, "cap": 1.0},
    {"case": "cpo_low40_o06_cap100", "mode": "low_sum", "threshold": 0.40, "overlay_target": 0.06, "cap": 1.0},
    {"case": "cpo_low40_o10_cap100", "mode": "low_sum", "threshold": 0.40, "overlay_target": 0.10, "cap": 1.0},
    {"case": "cpo_low60_o06_cap100", "mode": "low_sum", "threshold": 0.60, "overlay_target": 0.06, "cap": 1.0},
    {"case": "cpo_low60_o10_cap100", "mode": "low_sum", "threshold": 0.60, "overlay_target": 0.10, "cap": 1.0},
    {"case": "cpo_all_o03_cap100", "mode": "all", "overlay_target": 0.03, "cap": 1.0},
    {"case": "cpo_all_o06_cap100", "mode": "all", "overlay_target": 0.06, "cap": 1.0},
]


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    return df


def write_variant(core: pd.DataFrame, overlay: pd.DataFrame, case: dict) -> dict:
    base = core.copy()
    base["overlay_source"] = "core"
    existing_keys = set(zip(base["buy_date"].astype(str), base["stock_code"].astype(str)))
    daily_core_sum = base.groupby("buy_date")["target_pct"].sum()
    add = overlay.copy()
    add["overlay_source"] = "pullback_gap_overlay"
    add["target_pct"] = float(case["overlay_target"])
    add_keys = pd.Series(list(zip(add["buy_date"].astype(str), add["stock_code"].astype(str))), index=add.index)
    add = add[~add_keys.isin(existing_keys)].copy()
    if case["mode"] == "empty_only":
        core_days = set(base["buy_date"].astype(str).unique())
        add = add[~add["buy_date"].astype(str).isin(core_days)].copy()
    elif case["mode"] == "low_sum":
        threshold = float(case["threshold"])
        low_days = set(daily_core_sum[daily_core_sum < threshold].index.astype(str))
        add = add[add["buy_date"].astype(str).isin(low_days)].copy()
    elif case["mode"] == "all":
        pass
    else:
        raise ValueError(case["mode"])

    out = pd.concat([base, add], ignore_index=True)
    out = out.sort_values(["buy_date", "overlay_source", "rank"], ascending=[True, True, True]).copy()
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    out["target_pct"] = out["target_pct"] * (float(case["cap"]) / daily_sum).clip(upper=1.0)
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out_path = SIGNAL_DIR / f"{case['case']}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out_path),
        "mode": case["mode"],
        "overlay_target": float(case["overlay_target"]),
        "rows": int(len(out)),
        "core_rows": int(len(base)),
        "overlay_rows": int(len(add)),
        "buy_days": int(out["buy_date"].nunique()),
        "core_buy_days": int(base["buy_date"].nunique()),
        "overlay_buy_days": int(add["buy_date"].nunique()) if len(add) else 0,
        "stock_count": int(out["stock_code"].nunique()),
        "mean_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().max()),
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
    core = load_frame(CORE)
    overlay = load_frame(OVERLAY)
    manifest = [write_variant(core, overlay, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "overlay_rows": result.get("overlay_rows"),
                    "buy_days": result.get("buy_days"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(5).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"),
    }
    (REPORT_DIR / "core_plus_pullback_overlay_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_recalc_hardgate_scale_20260715.py"

spec = importlib.util.spec_from_file_location("rh_scale", BASE_SCRIPT)
rh = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(rh)

REPORT_DIR = rh.base.REPORT_DIR
SOURCE_AUDIT = REPORT_DIR / "recalc_hardgate_top1_audit_20260715.csv"
OUT_CSV = REPORT_DIR / "pass_ratio_recalc_conditional_scale_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_recalc_conditional_scale_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_recalc_conditional_scale_review_20260715.md"
SIGNAL_DIR = REPORT_DIR / "signals" / "recalc_conditional_scale"
LOG_DIR = REPORT_DIR / "logs" / "recalc_conditional_scale_20260715"

CASES = [
    {"case": "cond_base_s120", "base": 1.20, "cap": 0.65, "rules": []},
    {"case": "cond_deep_lowopen", "base": 1.20, "cap": 0.65, "rules": [("pct_le", -5.0, 1.12), ("gap_le", -0.8, 1.10)]},
    {"case": "cond_shallow_highgap_cut", "base": 1.20, "cap": 0.65, "rules": [("pct_gt", -2.5, 0.75), ("gap_gt", 0.5, 0.70)]},
    {"case": "cond_pred_extreme_cut", "base": 1.20, "cap": 0.65, "rules": [("pred5_ge", 0.998, 0.75), ("pred10_ge", 0.998, 0.80)]},
    {"case": "cond_turn_mid_boost", "base": 1.20, "cap": 0.65, "rules": [("turn_between", (3.0, 8.0), 1.12), ("turn_lt", 2.0, 0.85)]},
    {"case": "cond_amount_hi_boost", "base": 1.20, "cap": 0.65, "rules": [("amount_ge", 700000.0, 1.12), ("amount_lt", 180000.0, 0.90)]},
    {
        "case": "cond_combo_soft",
        "base": 1.20,
        "cap": 0.65,
        "rules": [("pct_le", -5.0, 1.08), ("gap_le", -0.8, 1.08), ("pct_gt", -2.5, 0.85), ("pred5_ge", 0.998, 0.85)],
    },
    {
        "case": "cond_combo_mid",
        "base": 1.20,
        "cap": 0.65,
        "rules": [("pct_le", -5.0, 1.12), ("gap_le", -0.8, 1.10), ("pct_gt", -2.5, 0.75), ("pred5_ge", 0.998, 0.75), ("turn_lt", 2.0, 0.85)],
    },
    {
        "case": "cond_combo_hi",
        "base": 1.25,
        "cap": 0.65,
        "rules": [("pct_le", -5.0, 1.10), ("gap_le", -0.8, 1.08), ("pct_gt", -2.5, 0.78), ("pred5_ge", 0.998, 0.80), ("gap_gt", 0.5, 0.80)],
    },
]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column), errors="coerce")


def apply_rules(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = source.copy()
    target = numeric(out, "target_pct").fillna(0.0) * float(case["base"])
    scale = pd.Series(1.0, index=out.index, dtype=float)
    for kind, threshold, factor in case["rules"]:
        if kind == "pct_le":
            mask = numeric(out, "signal_pct_chg_raw") <= float(threshold)
        elif kind == "pct_gt":
            mask = numeric(out, "signal_pct_chg_raw") > float(threshold)
        elif kind == "gap_le":
            mask = numeric(out, "exec_open_gap_pct") <= float(threshold)
        elif kind == "gap_gt":
            mask = numeric(out, "exec_open_gap_pct") > float(threshold)
        elif kind == "pred5_ge":
            mask = numeric(out, "pred_5d") >= float(threshold)
        elif kind == "pred10_ge":
            mask = numeric(out, "pred_10d") >= float(threshold)
        elif kind == "turn_lt":
            mask = numeric(out, "turnover_rate") < float(threshold)
        elif kind == "turn_between":
            low, high = threshold
            mask = numeric(out, "turnover_rate").between(float(low), float(high), inclusive="both")
        elif kind == "amount_ge":
            mask = numeric(out, "amount") >= float(threshold)
        elif kind == "amount_lt":
            mask = numeric(out, "amount") < float(threshold)
        else:
            raise ValueError(kind)
        scale = scale.mask(mask.fillna(False), scale * float(factor))
    out["target_pct_before_conditional_scale"] = numeric(out, "target_pct").fillna(0.0)
    out["conditional_scale_factor"] = scale
    out["target_pct"] = (target * scale).clip(lower=0.0, upper=float(case["cap"]))
    out = out[out["target_pct"] > 0].copy()
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["conditional_rules"] = json.dumps(case["rules"], ensure_ascii=False)
    return out


def run_case(signal_file: Path, case: str, slice_name: str) -> dict:
    old_log_dir = rh.base.LOG_DIR
    rh.base.LOG_DIR = LOG_DIR
    try:
        return rh.base.run_juejin(signal_file, case, slice_name)
    finally:
        rh.base.LOG_DIR = old_log_dir


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE_AUDIT, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    source = source[source["buy_day_hard_gate_complete_recalc"].astype(str).str.lower().isin(["true", "1"])].copy()
    rows: list[dict] = []
    for case in CASES:
        built = apply_rules(source, case)
        for slice_name, start in rh.base.SLICES:
            sliced = rh.base.slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_case(out_path, case["case"], slice_name)
            result.update(
                {
                    "base": float(case["base"]),
                    "cap": float(case["cap"]),
                    "rules": json.dumps(case["rules"], ensure_ascii=False),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                }
            )
            rows.append(result)
            print(json.dumps({"case": result["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    grouped = []
    for case, group in frame.groupby("case"):
        item = {"case": case}
        for sl in ["full", "from_202501", "recent60"]:
            row = group[group["slice"] == sl].iloc[0]
            item[f"{sl}_annual"] = row.get("pnl_ratio_annual")
            item[f"{sl}_sharpe"] = row.get("sharp_ratio")
            item[f"{sl}_mdd"] = row.get("max_drawdown")
        grouped.append(item)
    gframe = pd.DataFrame(grouped)
    gframe["min_annual"] = gframe[["full_annual", "from_202501_annual", "recent60_annual"]].min(axis=1)
    gframe["min_sharpe"] = gframe[["full_sharpe", "from_202501_sharpe", "recent60_sharpe"]].min(axis=1)
    gframe = gframe.sort_values(["min_annual", "min_sharpe"], ascending=False)

    lines = [
        "# 最新硬门槛 Top1 条件仓位调参掘金复跑 20260715",
        "",
        "## 说明",
        "",
        "- 本轮不补位、不换票，只按已知特征调节目标仓位。",
        "- 特征包括信号日跌幅、买入日未复权开盘缺口、成交额、换手率、5D/10D 分数极值。",
        "",
        "| 版本 | full年化 | full Sharpe | full回撤 | 2025以来年化 | recent60年化 | 三段最小年化 | 三段最小Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gframe.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {rh.base.pct(row['full_annual'])} | {float(row['full_sharpe']):.4f} | "
            f"{rh.base.pct(row['full_mdd'])} | {rh.base.pct(row['from_202501_annual'])} | "
            f"{rh.base.pct(row['recent60_annual'])} | {rh.base.pct(row['min_annual'])} | {float(row['min_sharpe']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

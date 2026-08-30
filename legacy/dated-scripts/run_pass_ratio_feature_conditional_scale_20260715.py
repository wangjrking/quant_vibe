from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_recalc_hardgate_scale_20260715.py"

spec = importlib.util.spec_from_file_location("recalc_scale", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE = REPORT_DIR / "recalc_hardgate_top1_audit_20260715.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "feature_conditional_scale"
LOG_DIR = REPORT_DIR / "logs" / "feature_conditional_scale_20260715"
OUT_CSV = REPORT_DIR / "feature_conditional_scale_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "feature_conditional_scale_juejin_20260715.json"
OUT_MD = REPORT_DIR / "feature_conditional_scale_review_20260715.md"

CASES = [
    {
        "case": "fc_s120_deep130_shallow085",
        "base_scale": 1.20,
        "cap": 0.65,
        "deep_scale": 1.30,
        "shallow_scale": 0.85,
        "gap_low_scale": 1.00,
        "gap_high_scale": 1.00,
        "amt_high_scale": 1.00,
        "pred_hi_scale": 1.00,
        "turn_mid_scale": 1.00,
    },
    {
        "case": "fc_s120_gaplow125_gaphi050",
        "base_scale": 1.20,
        "cap": 0.65,
        "deep_scale": 1.00,
        "shallow_scale": 1.00,
        "gap_low_scale": 1.25,
        "gap_high_scale": 0.50,
        "amt_high_scale": 1.00,
        "pred_hi_scale": 1.00,
        "turn_mid_scale": 1.00,
    },
    {
        "case": "fc_s120_amthi120_predhi070",
        "base_scale": 1.20,
        "cap": 0.65,
        "deep_scale": 1.00,
        "shallow_scale": 1.00,
        "gap_low_scale": 1.00,
        "gap_high_scale": 1.00,
        "amt_high_scale": 1.20,
        "pred_hi_scale": 0.70,
        "turn_mid_scale": 1.00,
    },
    {
        "case": "fc_s120_turnmid125_gaphi050",
        "base_scale": 1.20,
        "cap": 0.65,
        "deep_scale": 1.00,
        "shallow_scale": 1.00,
        "gap_low_scale": 1.00,
        "gap_high_scale": 0.50,
        "amt_high_scale": 1.00,
        "pred_hi_scale": 1.00,
        "turn_mid_scale": 1.25,
    },
    {
        "case": "fc_s125_combo_recent_proxy",
        "base_scale": 1.25,
        "cap": 0.65,
        "deep_scale": 1.20,
        "shallow_scale": 0.85,
        "gap_low_scale": 1.20,
        "gap_high_scale": 0.50,
        "amt_high_scale": 1.15,
        "pred_hi_scale": 0.75,
        "turn_mid_scale": 1.15,
    },
    {
        "case": "fc_s130_combo_strong",
        "base_scale": 1.30,
        "cap": 0.65,
        "deep_scale": 1.25,
        "shallow_scale": 0.75,
        "gap_low_scale": 1.25,
        "gap_high_scale": 0.35,
        "amt_high_scale": 1.20,
        "pred_hi_scale": 0.65,
        "turn_mid_scale": 1.20,
    },
]

SLICES = [
    ("full", None),
    ("from_202501", "20250102"),
    ("recent60", "__recent60__"),
]


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def build_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source[source["buy_day_hard_gate_complete_recalc"].astype(str).eq("True")].copy()
    pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    gap = pd.to_numeric(df["exec_open_gap_pct_recalc"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")
    pred10 = pd.to_numeric(df["pred_10d"], errors="coerce")
    turn = pd.to_numeric(df["turnover_rate"], errors="coerce")

    scale = pd.Series(float(case["base_scale"]), index=df.index)
    scale.loc[pct <= -5.0] *= float(case["deep_scale"])
    scale.loc[pct > -2.5] *= float(case["shallow_scale"])
    scale.loc[gap <= -1.0] *= float(case["gap_low_scale"])
    scale.loc[gap > 1.0] *= float(case["gap_high_scale"])
    scale.loc[amount >= 600000] *= float(case["amt_high_scale"])
    scale.loc[pred10 >= 0.999] *= float(case["pred_hi_scale"])
    scale.loc[turn.between(2.5, 6.5, inclusive="both")] *= float(case["turn_mid_scale"])

    df["target_pct_before_feature_scale"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["target_pct"] = (df["target_pct_before_feature_scale"] * scale).clip(upper=float(case["cap"]))
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["feature_conditional_scale"] = scale
    df["exec_open_gap_pct"] = df["exec_open_gap_pct_recalc"]
    df["buy_open_gap_raw_pct"] = df["exec_open_gap_pct_recalc"]
    df["buy_open_gap_pct"] = df["exec_open_gap_pct_recalc"]
    return df


def run_case(signal_file: Path, case_name: str, slice_name: str) -> dict:
    old_log_dir = base.base.LOG_DIR
    base.base.LOG_DIR = LOG_DIR
    try:
        return base.base.run_juejin(signal_file, case_name, slice_name)
    finally:
        base.base.LOG_DIR = old_log_dir


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows = []
    for case in CASES:
        built = build_case(source, case)
        for slice_name, start in SLICES:
            sliced = slice_frame(built, start)
            out = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out, index=False, encoding="utf-8-sig")
            result = run_case(out, case["case"], slice_name)
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result.update(
                {
                    **case,
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                    "signal_file": str(out),
                }
            )
            rows.append(result)
            print(json.dumps({"case": case["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio")}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 特征条件仓位缩放复核 20260715",
        "",
        "## 口径",
        "",
        "- 输入为最新 L2 重算硬门槛后的 Top1 信号。",
        "- 不增加信号、不补位，只按非日期特征调整仓位。",
        "- 特征来自信号日或买入日开盘可观察字段：信号日涨跌幅、买入日开盘缺口、成交额、换手率、模型分数。",
        "",
        "## 掘金结果",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 行数 | 买入日 | 均值仓位 | 最新买入日 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio') or 0):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(row['rows'])} | {int(row['buy_days'])} | {pct(row['mean_daily_target_sum'])} | {row['max_buy_date']} |"
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

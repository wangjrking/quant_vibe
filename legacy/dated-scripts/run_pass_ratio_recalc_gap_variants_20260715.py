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
SIGNAL_DIR = REPORT_DIR / "signals" / "recalc_gap_variants"
LOG_DIR = REPORT_DIR / "logs" / "recalc_gap_variants_20260715"
OUT_CSV = REPORT_DIR / "recalc_gap_variants_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "recalc_gap_variants_juejin_20260715.json"
OUT_MD = REPORT_DIR / "recalc_gap_variants_review_20260715.md"

CASES = [
    {"case": "rg_s120_all", "scale": 1.20, "cap": 0.65, "gap_max": None, "gap_downscale": None},
    {"case": "rg_s125_gap_le_1p0", "scale": 1.25, "cap": 0.65, "gap_max": 1.0, "gap_downscale": None},
    {"case": "rg_s120_gap_le_1p0", "scale": 1.20, "cap": 0.65, "gap_max": 1.0, "gap_downscale": None},
    {"case": "rg_s115_gap_le_1p0", "scale": 1.15, "cap": 0.65, "gap_max": 1.0, "gap_downscale": None},
    {"case": "rg_s120_gap_le_0p5", "scale": 1.20, "cap": 0.65, "gap_max": 0.5, "gap_downscale": None},
    {"case": "rg_s120_gap_le_0p0", "scale": 1.20, "cap": 0.65, "gap_max": 0.0, "gap_downscale": None},
    {"case": "rg_s125_gap_soft_gt1_x050", "scale": 1.25, "cap": 0.65, "gap_max": None, "gap_downscale": 0.50},
    {"case": "rg_s120_gap_soft_gt1_x030", "scale": 1.20, "cap": 0.65, "gap_max": None, "gap_downscale": 0.30},
]


def build_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source[source["buy_day_hard_gate_complete_recalc"].astype(str).eq("True")].copy()
    gap = pd.to_numeric(df["exec_open_gap_pct_recalc"], errors="coerce")
    if case["gap_max"] is not None:
        df = df[gap <= float(case["gap_max"])].copy()
        gap = pd.to_numeric(df["exec_open_gap_pct_recalc"], errors="coerce")
    raw = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0) * float(case["scale"])
    if case["gap_downscale"] is not None:
        raw.loc[gap > 1.0] *= float(case["gap_downscale"])
    df["target_pct_before_gap_variant"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    df["target_pct"] = raw.clip(upper=float(case["cap"]))
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["gap_variant_scale"] = float(case["scale"])
    df["gap_variant_cap"] = float(case["cap"])
    df["gap_variant_gap_max"] = case["gap_max"]
    df["gap_variant_gap_downscale"] = case["gap_downscale"]
    df["exec_open_gap_pct"] = df["exec_open_gap_pct_recalc"]
    df["buy_open_gap_raw_pct"] = df["exec_open_gap_pct_recalc"]
    df["buy_open_gap_pct"] = df["exec_open_gap_pct_recalc"]
    return df


def run_case(signal_file: Path, case_name: str) -> dict:
    old_log_dir = base.base.LOG_DIR
    base.base.LOG_DIR = LOG_DIR
    try:
        return base.base.run_juejin(signal_file, case_name, "full")
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
        df = build_case(source, case)
        out = SIGNAL_DIR / f"{case['case']}_full.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        result = run_case(out, case["case"])
        daily = df.groupby("buy_date")["target_pct"].sum() if len(df) else pd.Series(dtype=float)
        result.update(
            {
                **case,
                "signal_file": str(out),
                "rows": int(len(df)),
                "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
                "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                "max_buy_date": str(df["buy_date"].max()) if len(df) else None,
                "high_gap_rows": int((pd.to_numeric(df["exec_open_gap_pct"], errors="coerce") > 1.0).sum()) if len(df) else 0,
            }
        )
        rows.append(result)
        print(json.dumps({"case": result["case"], "rows": len(df), "max_buy": result["max_buy_date"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows).sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 最新核心 Top1 开盘涨幅收紧复核 20260715",
        "",
        "## 口径",
        "",
        "- 输入为已用最新 L2 未复权开盘价重算硬门槛的 Top1 信号。",
        "- 只测试买入日开盘涨幅门槛或高开降仓，不补位，不扩候选。",
        "- 正式收益指标以掘金输出为准。",
        "",
        "## 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 高开>1%行 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {int(row['buy_days'])} | {int(row.get('open_count') or 0)} | "
            f"{int(row['high_gap_rows'])} | {pct(row['mean_daily_target_sum'])} | {row['max_buy_date']} |"
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

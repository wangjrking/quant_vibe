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
SOURCE_DIR = REPORT_DIR / "signals" / "recalc_gap_variants"
SIGNAL_DIR = REPORT_DIR / "signals" / "recalc_gap_variant_slices"
LOG_DIR = REPORT_DIR / "logs" / "recalc_gap_variant_slices_20260715"
OUT_CSV = REPORT_DIR / "recalc_gap_variant_slices_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "recalc_gap_variant_slices_juejin_20260715.json"
OUT_MD = REPORT_DIR / "recalc_gap_variant_slices_review_20260715.md"

CASES = [
    "rg_s120_all",
    "rg_s125_gap_soft_gt1_x050",
    "rg_s120_gap_soft_gt1_x030",
    "rg_s115_gap_le_1p0",
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
    rows = []
    for case in CASES:
        source = SOURCE_DIR / f"{case}_full.csv"
        df = pd.read_csv(source, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
        for slice_name, start in SLICES:
            sliced = slice_frame(df, start)
            out = SIGNAL_DIR / f"{case}_{slice_name}.csv"
            sliced.to_csv(out, index=False, encoding="utf-8-sig")
            result = run_case(out, case, slice_name)
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result.update(
                {
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                    "signal_file": str(out),
                }
            )
            rows.append(result)
            print(json.dumps({"case": case, "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "rows": len(sliced)}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 最新核心开盘涨幅候选切片复核 20260715",
        "",
        "## 结果",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 行数 | 买入日 | 最新买入日 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio') or 0):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(row['rows'])} | {int(row['buy_days'])} | {row['max_buy_date']} |"
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

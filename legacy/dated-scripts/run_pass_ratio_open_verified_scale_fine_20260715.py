from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
)
SOURCE = REPORT_DIR / "signals" / "intended_top1" / "ogd_gap1_x050_open_verified_top1_full.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "open_verified_scale_fine"
LOG_DIR = REPORT_DIR / "logs" / "open_verified_scale_fine_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_open_verified_scale_fine_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_open_verified_scale_fine_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_open_verified_scale_fine_review_20260715.md"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

SLICES = [
    ("full", None),
    ("from_202501", "20250102"),
    ("recent60", "__recent60__"),
]
CASES = [
    {"case": "ov_top1_s105_cap055", "scale": 1.05, "cap": 0.55},
    {"case": "ov_top1_s110_cap055", "scale": 1.10, "cap": 0.55},
    {"case": "ov_top1_s115_cap055", "scale": 1.15, "cap": 0.55},
    {"case": "ov_top1_s120_cap055", "scale": 1.20, "cap": 0.55},
    {"case": "ov_top1_s105_cap060", "scale": 1.05, "cap": 0.60},
    {"case": "ov_top1_s110_cap060", "scale": 1.10, "cap": 0.60},
    {"case": "ov_top1_s115_cap060", "scale": 1.15, "cap": 0.60},
    {"case": "ov_top1_s120_cap060", "scale": 1.20, "cap": 0.60},
    {"case": "ov_top1_s125_cap060", "scale": 1.25, "cap": 0.60},
    {"case": "ov_top1_s130_cap060", "scale": 1.30, "cap": 0.60},
]


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def parse_metrics(text: str) -> dict:
    line = ""
    for item in reversed(text.splitlines()):
        if "GM_BACKTEST_INDICATOR:" in item:
            line = item
            break
    out: dict[str, float | int | str] = {"parse_status": "missing_indicator"}
    if not line:
        return out
    out["parse_status"] = "ok"
    for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "risk_ratio", "win_ratio", "calmar_ratio"]:
        match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", line)
        if match:
            out[key] = float(match.group(1))
    for key in ["open_count", "close_count", "win_count", "lose_count"]:
        match = re.search(rf"'{key}':\s*([0-9]+)", line)
        if match:
            out[key] = int(match.group(1))
    return out


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def build_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = source.copy()
    raw = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0) * float(case["scale"])
    out["target_pct_before_scale_fine"] = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    out["target_pct"] = raw.clip(upper=float(case["cap"]))
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["scale_fine_multiplier"] = float(case["scale"])
    out["scale_fine_cap"] = float(case["cap"])
    return out


def run_juejin(signal_file: Path, case: str, slice_name: str) -> dict:
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        "1",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {
        "case": case,
        "slice": slice_name,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "returncode": int(proc.returncode),
    }
    out.update(parse_metrics(text))
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows: list[dict] = []
    for case in CASES:
        built = build_case(source, case)
        for slice_name, start in SLICES:
            sliced = slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_juejin(out_path, case["case"], slice_name)
            result.update(
                {
                    "scale": float(case["scale"]),
                    "cap": float(case["cap"]),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                }
            )
            open_count = result.get("open_count")
            result["row_execution_ratio"] = float(open_count) / float(len(sliced)) if open_count is not None and len(sliced) else None
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
        "# 开盘验证 Top1 仓位微调掘金复跑 20260715",
        "",
        "## 结论",
        "",
        "- 本轮不补位、不扩候选，只对已通过开盘硬门槛的每日 Top1 做仓位乘数和单票上限微调。",
        "- 目标是提高后段收益，同时观察 full 口径 Sharpe 是否仍能保持在 4 以上。",
        "",
        "## 按三段最小年化排序",
        "",
        "| 版本 | full年化 | full Sharpe | 2025以来年化 | 2025以来Sharpe | recent60年化 | recent60 Sharpe | 三段最小年化 | 三段最小Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gframe.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row['full_annual'])} | {float(row['full_sharpe']):.4f} | "
            f"{pct(row['from_202501_annual'])} | {float(row['from_202501_sharpe']):.4f} | "
            f"{pct(row['recent60_annual'])} | {float(row['recent60_sharpe']):.4f} | "
            f"{pct(row['min_annual'])} | {float(row['min_sharpe']):.4f} |"
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

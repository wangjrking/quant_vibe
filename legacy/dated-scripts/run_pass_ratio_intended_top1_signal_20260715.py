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
SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "open_gap_deep_rebalance"
    / "ogd_gap1_x050.csv"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "intended_top1"
LOG_DIR = REPORT_DIR / "logs" / "intended_top1_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_intended_top1_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_intended_top1_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_intended_top1_review_20260715.md"

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


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def build_intended_top1(source: pd.DataFrame) -> pd.DataFrame:
    frame = source.copy()
    for col in ["target_pct", "sort_score", "rank", "pred_prob", "entry_score"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    sort_cols = [col for col in ["buy_date", "sort_score", "target_pct", "pred_prob", "entry_score", "stock_code"] if col in frame.columns]
    ascending = [True] + [False] * (len(sort_cols) - 2) + [True] if len(sort_cols) >= 2 else True
    frame = frame.sort_values(sort_cols, ascending=ascending)
    out = frame.groupby("buy_date", as_index=False).head(1).copy()
    out["strategy_variant"] = "ogd_gap1_x050_intended_top1"
    out["filter_name"] = "ogd_gap1_x050_intended_top1"
    out["intended_signal_scope"] = "daily_top1_only_no_refill_no_candidate_rows"
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    return out


def build_open_verified_top1(top1: pd.DataFrame) -> pd.DataFrame:
    frame = top1.copy()
    hard_complete = frame.get("buy_day_hard_gate_complete", pd.Series(dtype=object)).astype(str)
    frame = frame[hard_complete == "True"].copy()
    frame["strategy_variant"] = "ogd_gap1_x050_open_verified_top1"
    frame["filter_name"] = "ogd_gap1_x050_open_verified_top1"
    frame["intended_signal_scope"] = "daily_top1_open_verified_no_refill_no_candidate_rows"
    frame["daily_target_sum_after_cap"] = frame.groupby("buy_date")["target_pct"].transform("sum")
    return frame


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
    top1 = build_intended_top1(source)
    variants = [
        ("ogd_gap1_x050_intended_top1", top1),
        ("ogd_gap1_x050_open_verified_top1", build_open_verified_top1(top1)),
    ]
    rows: list[dict] = []
    for case_name, frame in variants:
        for slice_name, start in SLICES:
            sliced = slice_frame(frame, start)
            path = SIGNAL_DIR / f"{case_name}_{slice_name}.csv"
            sliced.to_csv(path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            hard_complete = sliced.get("buy_day_hard_gate_complete", pd.Series(dtype=object)).astype(str)
            result = run_juejin(path, case_name, slice_name)
            result.update(
                {
                    "source_rows": int(len(slice_frame(source, start))),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "stock_count": int(sliced["stock_code"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "hard_gate_true_rows": int((hard_complete == "True").sum()),
                    "hard_gate_false_rows": int((hard_complete == "False").sum()),
                    "hard_gate_nan_rows": int((hard_complete == "nan").sum()),
                }
            )
            open_count = result.get("open_count")
            result["row_execution_ratio"] = float(open_count) / float(len(sliced)) if open_count is not None and len(sliced) else None
            rows.append(result)
            print(json.dumps({"case": case_name, "slice": slice_name, "rows": len(sliced), "open": open_count, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 每日意图 Top1 信号通过率复核 20260715",
        "",
        "## 结论",
        "",
        "- 本轮不补位、不扩候选，只把原 `ogd_gap1_x050` 信号改为每日仅输出真正意图执行的 Top1。",
        "- 目的：提高正式信号文件的行级执行比例，避免把同日未执行的第二、第三行误解为交易信号。",
        "",
        "## 掘金复跑结果",
        "",
        "| 版本 | 切片 | 原行数 | 输出行数 | 开仓数 | 行级执行比例 | 年化 | Sharpe | 最大回撤 | 平均目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {int(row['source_rows'])} | {int(row['rows'])} | {int(row.get('open_count', 0))} | "
            f"{pct(row.get('row_execution_ratio'))} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | {pct(row.get('mean_daily_target_sum'))} |"
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

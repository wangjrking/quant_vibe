from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE = REPORT_DIR / "signals" / "feature_conditional_scale_fine" / "fcfine_s115_turnmid120_gaphi050_full.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "sell_frequency_fine"
LOG_DIR = REPORT_DIR / "logs" / "sell_frequency_fine_20260715"
OUT_CSV = REPORT_DIR / "sell_frequency_fine_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "sell_frequency_fine_juejin_20260715.json"
OUT_MD = REPORT_DIR / "sell_frequency_fine_review_20260715.md"

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
    {"case": "sf_base", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
    {"case": "sf_exit100", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 1.00, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
    {"case": "sf_exit102", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 1.02, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
    {"case": "sf_cont105", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.05, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
    {"case": "sf_mh2", "holding_days": 1, "max_holding_days": 2, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
    {"case": "sf_light2", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": 0.02, "light_min_hold": 1},
    {"case": "sf_light3", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": 0.03, "light_min_hold": 1},
    {"case": "sf_light5", "holding_days": 1, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": 0.05, "light_min_hold": 1},
    {"case": "sf_h2_mh3", "holding_days": 2, "max_holding_days": 3, "exit_ratio": 0.98, "continue_ratio": 1.02, "min_hold_score": 1, "light_stop": None, "light_min_hold": 0},
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
    out["holding_days"] = int(case["holding_days"])
    out["max_holding_days"] = int(case["max_holding_days"])
    out["score_exit_entry_ratio"] = float(case["exit_ratio"])
    out["score_continue_entry_ratio"] = float(case["continue_ratio"])
    out["min_holding_days_before_score_exit"] = int(case["min_hold_score"])
    out["strategy_variant"] = f"fcfine_s115_turnmid120_gaphi050_{case['case']}"
    out["filter_name"] = f"fcfine_s115_turnmid120_gaphi050_{case['case']}"
    return out


def run_juejin(signal_file: Path, case: dict, slice_name: str) -> dict:
    log_file = LOG_DIR / f"{case['case']}_{slice_name}.log"
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
        "--holding-days",
        str(case["holding_days"]),
        "--max-holding-days",
        str(case["max_holding_days"]),
        "--score-exit-entry-ratio",
        str(case["exit_ratio"]),
        "--score-continue-entry-ratio",
        str(case["continue_ratio"]),
        "--min-holding-days-before-score-exit",
        str(case["min_hold_score"]),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    if case["light_stop"] is not None:
        cmd.extend(["--light-stop-loss-pct", str(case["light_stop"])])
        cmd.extend(["--min-holding-days-before-light-stop", str(case["light_min_hold"])])
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"case": case["case"], "slice": slice_name, "signal_file": str(signal_file), "log_file": str(log_file), "returncode": int(proc.returncode)}
    out.update(parse_metrics(text))
    if out.get("parse_status") != "ok":
        out["stdout_tail"] = proc.stdout[-1000:]
        out["stderr_tail"] = proc.stderr[-1000:]
    return out


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


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
            result = run_juejin(out_path, case, slice_name)
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result.update(
                {
                    **case,
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                }
            )
            rows.append(result)
            print(json.dumps({"case": result["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 卖出频率精修复核 20260715",
        "",
        "## 口径",
        "",
        "- 输入为 `fcfine_s115_turnmid120_gaphi050`，只调整卖出频率和止损，不改变买入信号。",
        "- 正式收益指标以掘金输出为准。",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 平仓 | 行数 | 买入日 | 最新买入日 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio') or 0):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{int(row.get('open_count') or 0)} | {int(row.get('close_count') or 0)} | "
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

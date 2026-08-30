from __future__ import annotations

import ast
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
STRATEGY_DIR = REPORT_DIR / "code_snapshots" / "replacement_exit"
SIGNAL_DIR = REPORT_DIR / "signals" / "replacement_exit"
LOG_DIR = REPORT_DIR / "logs" / "replacement_exit_20260715"
OUT_CSV = REPORT_DIR / "replacement_exit_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "replacement_exit_juejin_20260715.json"
OUT_MD = REPORT_DIR / "replacement_exit_review_20260715.md"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

SIGNALS = {
    "frs65": REPORT_DIR / "signals" / "full_refill_scale_smooth" / "frs_scale065_cap070_full.csv",
    "frb_q105": REPORT_DIR / "signals" / "full_refill_bucket_smooth" / "frb_q2_100_q1_050_q0_020_rf015_full.csv",
    "rank_overlay": REPORT_DIR / "signals" / "rank_overlay_current_core" / "core_frs65_r10_985_r1_900_broad_t1_lowsum.csv",
}

CASES = [
    {"case": "h2_mh4_min1_r100_e000", "hold": 2, "max_hold": 4, "min_hold": 1, "ratio": 1.00, "edge": 0.000},
    {"case": "h2_mh4_min1_r101_e000", "hold": 2, "max_hold": 4, "min_hold": 1, "ratio": 1.01, "edge": 0.000},
    {"case": "h2_mh4_min1_r102_e002", "hold": 2, "max_hold": 4, "min_hold": 1, "ratio": 1.02, "edge": 0.002},
    {"case": "h2_mh4_min1_r103_e003", "hold": 2, "max_hold": 4, "min_hold": 1, "ratio": 1.03, "edge": 0.003},
    {"case": "h3_mh5_min1_r100_e000", "hold": 3, "max_hold": 5, "min_hold": 1, "ratio": 1.00, "edge": 0.000},
    {"case": "h3_mh5_min1_r101_e000", "hold": 3, "max_hold": 5, "min_hold": 1, "ratio": 1.01, "edge": 0.000},
    {"case": "h3_mh5_min1_r102_e002", "hold": 3, "max_hold": 5, "min_hold": 1, "ratio": 1.02, "edge": 0.002},
    {"case": "h3_mh5_min1_r103_e003", "hold": 3, "max_hold": 5, "min_hold": 1, "ratio": 1.03, "edge": 0.003},
    {"case": "h3_mh5_min2_r100_e000", "hold": 3, "max_hold": 5, "min_hold": 2, "ratio": 1.00, "edge": 0.000},
    {"case": "h3_mh5_min2_r101_e000", "hold": 3, "max_hold": 5, "min_hold": 2, "ratio": 1.01, "edge": 0.000},
    {"case": "h3_mh5_min2_r102_e002", "hold": 3, "max_hold": 5, "min_hold": 2, "ratio": 1.02, "edge": 0.002},
    {"case": "h4_mh6_min1_r100_e000", "hold": 4, "max_hold": 6, "min_hold": 1, "ratio": 1.00, "edge": 0.000},
    {"case": "h4_mh6_min1_r101_e000", "hold": 4, "max_hold": 6, "min_hold": 1, "ratio": 1.01, "edge": 0.000},
    {"case": "h4_mh6_min1_r102_e002", "hold": 4, "max_hold": 6, "min_hold": 1, "ratio": 1.02, "edge": 0.002},
]


def parse_metrics(text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    payload = None
    for line in reversed(text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            break
    if not payload:
        return {"parse_status": "missing_indicator"}
    try:
        indicator = ast.literal_eval(payload)
        out = {"parse_status": "ok"}
        out.update(indicator)
        return out
    except Exception:
        out = {"parse_status": "regex_fallback"}
        for key in ["pnl_ratio", "pnl_ratio_annual", "sharp_ratio", "max_drawdown", "risk_ratio", "win_ratio", "calmar_ratio"]:
            match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
            if match:
                out[key] = float(match.group(1))
        for key in ["open_count", "close_count", "win_count", "lose_count"]:
            match = re.search(rf"'{key}':\s*([0-9]+)", payload)
            if match:
                out[key] = int(match.group(1))
        return out


def infer_backtest_window(signal_file: Path, holding_days: int) -> tuple[str, str]:
    buy_dates = []
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            value = str(row.get("buy_date") or "").strip()
            if value:
                buy_dates.append(datetime.strptime(value, "%Y%m%d"))
    if not buy_dates:
        raise RuntimeError(f"No buy_date in {signal_file}")
    return (
        min(buy_dates).strftime("%Y-%m-%d 09:00:00"),
        (max(buy_dates) + timedelta(days=max(holding_days * 3, 10))).strftime("%Y-%m-%d 15:30:00"),
    )


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def num(value: object, default: float = 0.0) -> float:
    if value is None or value == "" or pd.isna(value):
        return default
    return float(value)


def integer(value: object, default: int = 0) -> int:
    if value is None or value == "" or pd.isna(value):
        return default
    return int(float(value))


def prepare_signal(signal_name: str, source_file: Path, case: dict) -> tuple[Path, pd.DataFrame]:
    df = pd.read_csv(source_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["holding_days"] = int(case["hold"])
    df["max_holding_days"] = int(case["max_hold"])
    df["score_exit_entry_ratio"] = 0.0
    df["signal_score_exit_entry_ratio"] = 0.0
    df["score_continue_entry_ratio"] = 9.99
    df["signal_score_continue_entry_ratio"] = 9.99
    df["min_holding_days_before_score_exit"] = 99
    df["signal_min_holding_days_before_score_exit"] = 99
    df["strategy_variant"] = f"{signal_name}_{case['case']}"
    df["filter_name"] = f"{signal_name}_{case['case']}"
    out_path = SIGNAL_DIR / f"{signal_name}_{case['case']}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, df


def run_juejin(signal_file: Path, log_file: Path, max_positions: int, case: dict) -> dict:
    start, end = infer_backtest_window(signal_file, int(case["hold"]))
    env = os.environ.copy()
    env.update(
        {
            "GM_REPLACE_EXIT_MODE": "1",
            "GM_REPLACE_EXIT_MIN_HOLDING_DAYS": str(case["min_hold"]),
            "GM_REPLACE_EXIT_SCORE_RATIO": str(case["ratio"]),
            "GM_REPLACE_EXIT_MIN_EDGE": str(case["edge"]),
        }
    )
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
        str(max_positions),
        "--holding-days",
        str(case["hold"]),
        "--max-holding-days",
        str(case["max_hold"]),
        "--score-exit-entry-ratio",
        "0",
        "--score-continue-entry-ratio",
        "9.99",
        "--min-holding-days-before-score-exit",
        "99",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        start,
        "--backtest-end",
        end,
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), env=env, capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"returncode": int(proc.returncode), "log_file": str(log_file), "backtest_start": start, "backtest_end": end}
    out.update(parse_metrics(text))
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for signal_name, source_file in SIGNALS.items():
        for case in CASES:
            signal_file, signal = prepare_signal(signal_name, source_file, case)
            max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
            log_file = LOG_DIR / f"{signal_name}_{case['case']}_mp{max_positions}.log"
            result = run_juejin(signal_file, log_file, max_positions, case)
            daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
            result.update(
                {
                    "case": f"{signal_name}_{case['case']}",
                    "signal_name": signal_name,
                    **case,
                    "signal_file": str(signal_file),
                    "rows": int(len(signal)),
                    "buy_days": int(signal["buy_date"].nunique()) if len(signal) else 0,
                    "stock_count": int(signal["stock_code"].nunique()) if len(signal) else 0,
                    "max_positions": max_positions,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(signal["buy_date"].max()) if len(signal) else None,
                }
            )
            rows.append(result)
            print(json.dumps({"case": result["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "buy_days": result["buy_days"]}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    for column in ["sharp_ratio", "pnl_ratio_annual", "max_drawdown"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 替换式卖出规则验证 20260715",
        "",
        "## 口径",
        "",
        "- 使用 research-only 掘金代码快照，不修改生产策略。",
        "- 卖出规则：当日新信号分数明显高于持仓票当前 formal 10D 分数时，先卖旧持仓，再由买入流程买新信号。",
        "- 该规则比较当前持仓分数与当日新买入信号分数，不依赖历史买入分数。",
        "- 本轮指标以掘金日志为准。",
        "",
        "## full 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 平仓 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {num(row.get('sharp_ratio')):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {integer(row.get('buy_days'))} | {integer(row.get('open_count'))} | "
            f"{integer(row.get('close_count'))} | {pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
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
            f"- research-only 代码快照：`{STRATEGY_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

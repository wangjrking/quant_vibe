from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "top3_open_quality_pick1"
LOG_DIR = REPORT_DIR / "logs" / "pick1_slice_stability"
OUT_CSV = REPORT_DIR / "pick1_slice_stability_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "pick1_slice_stability_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "pick1候选切片稳定性复核_20260714.md"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


CASES = [
    "pick1_mid_deep_only_quality_pred10_h91_m35_l05",
    "pick1_mid_deep_only_quality_pred10_h85_m35_l10",
    "pick1_mid_atr12_quality_pred10_h91_m35_l05",
    "pick1_mid_gap_amt_quality_pred10_h91_m35_l05",
]

PERIODS = [
    ("full", "2022-08-12 09:00:00", "2026-07-24 15:30:00"),
    ("2024", "2024-01-01 09:00:00", "2024-12-31 15:30:00"),
    ("2025", "2025-01-01 09:00:00", "2025-12-31 15:30:00"),
    ("2026ytd", "2026-01-01 09:00:00", "2026-07-24 15:30:00"),
    ("from_202501", "2025-01-01 09:00:00", "2026-07-24 15:30:00"),
    ("from_202407", "2024-07-01 09:00:00", "2026-07-24 15:30:00"),
]


def extract_indicator(text: str) -> dict:
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(text.splitlines()):
        if marker in line:
            payload = line.split(marker, 1)[1].strip()
            try:
                return ast.literal_eval(payload)
            except Exception:
                out: dict[str, float | int | str] = {"raw_indicator": payload}
                for key in [
                    "pnl_ratio",
                    "pnl_ratio_annual",
                    "sharp_ratio",
                    "max_drawdown",
                    "risk_ratio",
                    "win_ratio",
                    "calmar_ratio",
                ]:
                    match = re.search(rf"'{key}':\s*([-+0-9.eE]+)", payload)
                    if match:
                        out[key] = float(match.group(1))
                for key in ["open_count", "close_count", "win_count", "lose_count"]:
                    match = re.search(rf"'{key}':\s*([0-9]+)", payload)
                    if match:
                        out[key] = int(match.group(1))
                return out
    return {}


def run_case(case: str, period: str, start: str, end: str) -> dict:
    signal_file = SIGNAL_DIR / f"{case}.csv"
    log_file = LOG_DIR / f"{case}_{period}_slip0p0030.log"
    base = {
        "case": case,
        "period": period,
        "start": start,
        "end": end,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "slippage": 0.003,
    }
    if log_file.exists() and log_file.stat().st_size > 0:
        indicator = extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore"))
        if indicator.get("pnl_ratio_annual") is not None:
            return {**base, "returncode": 0, **indicator}
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
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
        "--backtest-slippage-ratio",
        "0.0030",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    indicator = extract_indicator(log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "")
    out = {**base, "returncode": proc.returncode}
    if indicator:
        out.update(indicator)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def signal_audit(case: str) -> dict:
    path = SIGNAL_DIR / f"{case}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    day = df.groupby("buy_date")["target_pct"].sum()
    return {
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "min_signal_date": str(df["signal_date"].min()),
        "max_signal_date": str(df["signal_date"].max()),
        "min_buy_date": str(df["buy_date"].min()),
        "max_buy_date": str(df["buy_date"].max()),
        "mean_target_sum": float(day.mean()) if len(day) else 0.0,
        "max_target_sum": float(day.max()) if len(day) else 0.0,
        "duplicate_buy_stock_keys": int(df.groupby(["buy_date", "stock_code"]).size().gt(1).sum()),
        "bj_rows": int(df["stock_code"].astype(str).str.endswith(".BJ").sum()),
    }


def write_report(frame: pd.DataFrame, audits: dict[str, dict]) -> None:
    def num(value, default=0.0) -> float:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float(default)
        return float(value)

    lines = [
        "# pick1 候选切片稳定性复核",
        "",
        "## 当前结论",
        "",
        "- 本轮只复核从当前生产 full_history 信号派生的 pick1 候选。",
        "- 回测显式使用 0.30% 滑点，和当前生产 validation 口径一致。",
        "- 结论仍以掘金日志为准；本报告不发布生产策略。",
        "",
        "## 掘金结果",
        "",
        "| 候选 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 胜率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["case", "period"]).iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('period')} | "
            f"{num(row.get('pnl_ratio_annual')) * 100:.2f}% | "
            f"{num(row.get('sharp_ratio')):.3f} | "
            f"{num(row.get('max_drawdown')) * 100:.2f}% | "
            f"{int(num(row.get('open_count')))} | "
            f"{num(row.get('win_ratio')) * 100:.2f}% |"
        )
    lines.extend(["", "## 信号审计", "", "| 候选 | 行数 | 买入日 | 股票数 | 日期覆盖 | 平均目标仓位 | BJ | 重复键 |", "|---|---:|---:|---:|---|---:|---:|---:|"])
    for case, audit in audits.items():
        lines.append(
            f"| {case} | {audit['rows']} | {audit['buy_days']} | {audit['stock_count']} | "
            f"{audit['min_signal_date']} 到 {audit['max_signal_date']} | "
            f"{audit['mean_target_sum'] * 100:.2f}% | {audit['bj_rows']} | {audit['duplicate_buy_stock_keys']} |"
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
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    audits = {case: signal_audit(case) for case in CASES}
    results = []
    for case in CASES:
        for period, start, end in PERIODS:
            row = run_case(case, period, start, end)
            results.append(row)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
            OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "case": case,
                        "period": period,
                        "annual": row.get("pnl_ratio_annual"),
                        "sharpe": row.get("sharp_ratio"),
                        "mdd": row.get("max_drawdown"),
                        "open": row.get("open_count"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame, audits)
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

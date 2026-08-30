from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_full_refill_scale_smooth_20260715.py"

spec = importlib.util.spec_from_file_location("smooth_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SIGNAL_DIR = REPORT_DIR / "signals" / "independent_rank_exit"
LOG_DIR = REPORT_DIR / "logs" / "independent_rank_exit_20260715"
OUT_CSV = REPORT_DIR / "independent_rank_exit_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "independent_rank_exit_juejin_20260715.json"
OUT_MD = REPORT_DIR / "independent_rank_exit_review_20260715.md"

SIGNALS = {
    "frs65": REPORT_DIR / "signals" / "full_refill_scale_smooth" / "frs_scale065_cap070_full.csv",
    "frb_q105": REPORT_DIR / "signals" / "full_refill_bucket_smooth" / "frb_q2_100_q1_050_q0_020_rf015_full.csv",
    "rank_overlay": REPORT_DIR / "signals" / "rank_overlay_current_core" / "core_frs65_r10_985_r1_900_broad_t1_lowsum.csv",
}

CASES = []
for hold in [2, 3, 4]:
    for rank in [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95]:
        CASES.append({"case": f"h{hold}_rank{int(rank*100):03d}", "holding_days": hold, "max_holding_days": hold, "score_exit_rank": rank})


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


def prepare_signal(source_file: Path, signal_name: str, case: dict) -> tuple[Path, pd.DataFrame]:
    df = pd.read_csv(source_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["holding_days"] = int(case["holding_days"])
    df["max_holding_days"] = int(case["max_holding_days"])
    # Disable entry-relative exit so the tested sell rule is independent:
    # it only depends on current daily 10D rank from the active formal score table.
    df["score_exit_entry_ratio"] = 0.0
    df["signal_score_exit_entry_ratio"] = 0.0
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = 9.99
    df["signal_score_continue_entry_ratio"] = 9.99
    df["strategy_variant"] = f"{signal_name}_{case['case']}"
    df["filter_name"] = f"{signal_name}_{case['case']}"
    out_path = SIGNAL_DIR / f"{signal_name}_{case['case']}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path, df


def run_case(signal_file: Path, log_file: Path, max_positions: int, case: dict) -> dict:
    cmd = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir",
        str(base.STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(max_positions),
        "--holding-days",
        str(case["holding_days"]),
        "--max-holding-days",
        str(case["max_holding_days"]),
        "--score-exit-entry-ratio",
        "0",
        "--score-exit-rank",
        str(case["score_exit_rank"]),
        "--min-holding-days-before-score-exit",
        "1",
        "--score-continue-entry-ratio",
        "9.99",
        "--score-db",
        str(base.SCORE_DB),
        "--score-table",
        base.SCORE_TABLE,
        "--market-db",
        str(base.MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"returncode": int(proc.returncode), "log_file": str(log_file)}
    out.update(base.parse_metrics(text))
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for signal_name, source_file in SIGNALS.items():
        for case in CASES:
            signal_file, signal = prepare_signal(source_file, signal_name, case)
            max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
            log_file = LOG_DIR / f"{signal_name}_{case['case']}_mp{max_positions}.log"
            result = run_case(signal_file, log_file, max_positions, case)
            daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
            result.update(
                {
                    "signal_name": signal_name,
                    **case,
                    "case": f"{signal_name}_{case['case']}",
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
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 独立排名卖出规则验证 20260715",
        "",
        "## 口径",
        "",
        "- 输入：当前覆盖到 20260714 的高覆盖信号。",
        "- 卖出规则：持仓期间若当前 10D formal 分数日截面 rank 跌出阈值，则次日开盘卖出。",
        "- 该卖出规则不依赖入场分数，仅依赖当前持仓票的最新 formal 10D rank。",
        "- 本轮为 research-only，指标以掘金日志为准。",
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
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

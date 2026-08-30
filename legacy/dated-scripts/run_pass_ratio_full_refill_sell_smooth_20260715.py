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
LOG_DIR = REPORT_DIR / "logs" / "full_refill_sell_smooth_20260715"
OUT_CSV = REPORT_DIR / "full_refill_sell_smooth_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "full_refill_sell_smooth_juejin_20260715.json"
OUT_MD = REPORT_DIR / "full_refill_sell_smooth_review_20260715.md"

SIGNALS = {
    "frs_scale065_cap070": REPORT_DIR / "signals" / "full_refill_scale_smooth" / "frs_scale065_cap070_full.csv",
    "frb_q2_100_q1_030_q0_000_rf015": REPORT_DIR / "signals" / "full_refill_bucket_smooth" / "frb_q2_100_q1_030_q0_000_rf015_full.csv",
    "frb_q2_100_q1_050_q0_020_rf015": REPORT_DIR / "signals" / "full_refill_bucket_smooth" / "frb_q2_100_q1_050_q0_020_rf015_full.csv",
}
SELL_CASES = [
    {"sell": "mh1_e098_c999", "max_holding_days": 1, "score_exit_entry_ratio": 0.98, "score_continue_entry_ratio": 9.99},
    {"sell": "mh1_e100_c999", "max_holding_days": 1, "score_exit_entry_ratio": 1.00, "score_continue_entry_ratio": 9.99},
    {"sell": "mh2_e098_c999", "max_holding_days": 2, "score_exit_entry_ratio": 0.98, "score_continue_entry_ratio": 9.99},
    {"sell": "mh2_e100_c999", "max_holding_days": 2, "score_exit_entry_ratio": 1.00, "score_continue_entry_ratio": 9.99},
    {"sell": "mh2_e098_c102", "max_holding_days": 2, "score_exit_entry_ratio": 0.98, "score_continue_entry_ratio": 1.02},
    {"sell": "mh3_e100_c999", "max_holding_days": 3, "score_exit_entry_ratio": 1.00, "score_continue_entry_ratio": 9.99},
]


def run_case(signal_file: Path, name: str, sell_case: dict, max_positions: int) -> dict:
    log_file = LOG_DIR / f"{name}_{sell_case['sell']}.log"
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
        "1",
        "--max-holding-days",
        str(sell_case["max_holding_days"]),
        "--score-exit-entry-ratio",
        str(sell_case["score_exit_entry_ratio"]),
        "--score-continue-entry-ratio",
        str(sell_case["score_continue_entry_ratio"]),
        "--min-holding-days-before-score-exit",
        "1",
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


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, signal_file in SIGNALS.items():
        signal = pd.read_csv(signal_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
        max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
        daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
        for sell_case in SELL_CASES:
            result = run_case(signal_file, name, sell_case, max_positions)
            result.update(
                {
                    "case": f"{name}_{sell_case['sell']}",
                    "signal_name": name,
                    **sell_case,
                    "signal_file": str(signal_file),
                    "rows": int(len(signal)),
                    "buy_days": int(signal["buy_date"].nunique()) if len(signal) else 0,
                    "max_positions": max_positions,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_buy_date": str(signal["buy_date"].max()) if len(signal) else None,
                }
            )
            rows.append(result)
            print(json.dumps({"case": result["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 完整高覆盖信号卖出频率平滑验证 20260715",
        "",
        "## full 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 平仓 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {int(row.get('buy_days') or 0)} | "
            f"{int(row.get('open_count') or 0)} | {int(row.get('close_count') or 0)} | {row.get('max_buy_date')} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

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
SIGNAL_DIR = REPORT_DIR / "signals" / "sell_rule_grid"
LOG_DIR = REPORT_DIR / "logs" / "sell_rule_grid_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_sell_rule_grid_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_sell_rule_grid_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_sell_rule_grid_review_20260715.md"

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
RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = (
    MAIN
    / "strategy_library"
    / "production"
    / "prod_fw_soft_deepdrop_weight_v20260706"
    / "code_snapshot"
)
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

CASES = [
    {"case": "sell_base_h1_mh3_e098_c102", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh2_e098_c102", "h": 1, "mh": 2, "exit": 0.98, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh4_e098_c102", "h": 1, "mh": 4, "exit": 0.98, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh3_e095_c102", "h": 1, "mh": 3, "exit": 0.95, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh3_e100_c102", "h": 1, "mh": 3, "exit": 1.00, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh3_e098_c100", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.00, "minh": 1},
    {"case": "sell_h1_mh3_e098_c105", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.05, "minh": 1},
    {"case": "sell_h2_mh3_e098_c102", "h": 2, "mh": 3, "exit": 0.98, "cont": 1.02, "minh": 1},
    {"case": "sell_h2_mh4_e098_c102", "h": 2, "mh": 4, "exit": 0.98, "cont": 1.02, "minh": 1},
    {"case": "sell_h1_mh3_e098_c102_minh2", "h": 1, "mh": 3, "exit": 0.98, "cont": 1.02, "minh": 2},
]
SLICES = [("full", None)]


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


def build_signal(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = source.copy()
    out["holding_days"] = int(case["h"])
    out["max_holding_days"] = int(case["mh"])
    out["score_exit_entry_ratio"] = float(case["exit"])
    out["min_holding_days_before_score_exit"] = int(case["minh"])
    out["score_continue_entry_ratio"] = float(case["cont"])
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    return out


def run_juejin(path: Path, case: str, slice_name: str) -> dict:
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(path),
        "--log-file",
        str(log_file),
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"case": case, "slice": slice_name, "signal_file": str(path), "log_file": str(log_file), "returncode": int(proc.returncode)}
    out.update(parse_metrics(text))
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows: list[dict] = []
    for case in CASES:
        built = build_signal(source, case)
        path = SIGNAL_DIR / f"{case['case']}_full.csv"
        built.to_csv(path, index=False, encoding="utf-8-sig")
        result = run_juejin(path, case["case"], "full")
        result.update(case)
        result.update(
            {
                "rows": int(len(built)),
                "buy_days": int(built["buy_date"].nunique()),
                "mean_daily_target_sum": float(built.groupby("buy_date")["target_pct"].sum().mean()),
                "max_daily_target_sum": float(built.groupby("buy_date")["target_pct"].sum().max()),
            }
        )
        rows.append(result)
        print(
            json.dumps(
                {
                    "case": result["case"],
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open_count": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 同源信号卖出频率参数 full 掘金复跑 20260715",
        "",
        "| 版本 | h | mh | exit | continue | min_hold | 年化 | Sharpe | 最大回撤 | 开仓 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['h'])} | {int(row['mh'])} | {float(row['exit']):.2f} | "
            f"{float(row['cont']):.2f} | {int(row['minh'])} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | {int(row.get('open_count', 0))} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

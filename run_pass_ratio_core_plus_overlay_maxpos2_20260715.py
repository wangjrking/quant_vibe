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
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_daily_top1_overlay"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_daily_top1_overlay_maxpos2_20260715"
OUT_CSV = REPORT_DIR / "core_plus_daily_top1_overlay_maxpos2_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "core_plus_daily_top1_overlay_maxpos2_juejin_20260715.json"
OUT_MD = REPORT_DIR / "core_plus_daily_top1_overlay_maxpos2_review_20260715.md"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

CASES = [
    "core_only_ref",
    "core_plus_p150pos_x025",
    "core_plus_p150pos_x050",
    "core_plus_alltiny_x025",
    "core_plus_alltiny_x050",
    "core_plus_rankguard_x025",
    "core_plus_rankguard_x050",
]
SLICES = ["full", "from_202501", "recent60"]


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
        ind = ast.literal_eval(payload)
        out = {"parse_status": "ok"}
        out.update(ind)
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


def run_juejin(signal_file: Path, log_file: Path) -> dict:
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
        "2",
        "--score-db",
        str(SCORE_DB),
        "--score-table",
        SCORE_TABLE,
        "--market-db",
        str(MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    out = {"returncode": int(proc.returncode), "log_file": str(log_file)}
    out.update(parse_metrics(text))
    return out


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for case in CASES:
        for slice_name in SLICES:
            signal_file = SIGNAL_DIR / f"{case}_{slice_name}.csv"
            if not signal_file.exists():
                continue
            signal = pd.read_csv(signal_file, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
            log_file = LOG_DIR / f"{case}_{slice_name}.log"
            result = run_juejin(signal_file, log_file)
            overlay_rows = int((signal.get("overlay_source", pd.Series(dtype=str)).astype(str) != "core").sum()) if len(signal) else 0
            daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
            result.update(
                {
                    "case": case,
                    "slice": slice_name,
                    "signal_file": str(signal_file),
                    "rows": int(len(signal)),
                    "buy_days": int(signal["buy_date"].nunique()) if len(signal) else 0,
                    "stock_count": int(signal["stock_code"].nunique()) if len(signal) else 0,
                    "overlay_rows": overlay_rows,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(signal["buy_date"].max()) if len(signal) else None,
                    "max_positions": 2,
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        "case": case,
                        "slice": slice_name,
                        "buy_days": result["buy_days"],
                        "annual": result.get("pnl_ratio_annual"),
                        "sharpe": result.get("sharp_ratio"),
                        "mdd": result.get("max_drawdown"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    full = frame[frame["slice"].eq("full")].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 核心信号叠加每日 Top1 小仓位双持仓验证 20260715",
        "",
        "## 口径",
        "",
        "- 信号文件复用核心信号 + 弱通过层，不新增候补池。",
        "- 掘金回测参数改为 `max_positions=2`，避免弱通过层占用单票路径后挡住核心主仓。",
        "- 本轮为 research-only，未修改生产策略。",
        "",
        "## full 结果",
        "",
        "| 版本 | 买入日 | overlay行 | 年化 | Sharpe | 最大回撤 | 胜率 | 开仓 | 平仓 | 均值目标仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in full.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['buy_days'])} | {int(row['overlay_rows'])} | "
            f"{pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {pct(row.get('win_ratio'))} | "
            f"{int(row.get('open_count') or 0)} | {int(row.get('close_count') or 0)} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 输入信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

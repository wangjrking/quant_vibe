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
SIGNAL_DIR = REPORT_DIR / "signals" / "position_utilization"
LOG_DIR = REPORT_DIR / "logs" / "position_utilization_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_position_utilization_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_position_utilization_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_position_utilization_review_20260715.md"

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

TARGETS = [0.70, 0.85, 1.00]
SLICES = [
    ("full", None),
    ("from_202407", "20240701"),
    ("from_202501", "20250102"),
    ("recent60", "__recent60__"),
]


def pct(value: object, digits: int = 2) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def parse_metrics(log_text: str) -> dict:
    line = ""
    for item in reversed(log_text.splitlines()):
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


def slice_frame(frame: pd.DataFrame, slice_name: str, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def scale_to_target(frame: pd.DataFrame, target: float) -> pd.DataFrame:
    out = frame.copy()
    raw = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    daily_sum = raw.groupby(out["buy_date"]).transform("sum")
    scale = (target / daily_sum).where(daily_sum > 0, 0.0)
    scale = scale.clip(lower=0.0, upper=target / 1e-9)
    out["target_pct_before_utilization"] = raw
    out["target_pct"] = (raw * scale).clip(upper=0.90)
    # A per-stock cap can pull the daily sum below target; never exceed 100%.
    daily_after = out["target_pct"].groupby(out["buy_date"]).transform("sum")
    cap_scale = (1.0 / daily_after).where(daily_after > 1.0, 1.0)
    out["target_pct"] = out["target_pct"] * cap_scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = f"ogd_gap1_x050_util{int(target * 100)}"
    out["filter_name"] = f"ogd_gap1_x050_util{int(target * 100)}"
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
        "stdout_tail": (proc.stdout or "")[-1000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
    }
    out.update(parse_metrics(text))
    return out


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows: list[dict] = []
    for target in TARGETS:
        scaled = scale_to_target(source, target)
        case = f"ogd_gap1_x050_util{int(target * 100)}"
        for slice_name, start in SLICES:
            sliced = slice_frame(scaled, slice_name, start)
            path = SIGNAL_DIR / f"{case}_{slice_name}.csv"
            sliced.to_csv(path, index=False, encoding="utf-8-sig")
            daily_sum = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_juejin(path, case, slice_name)
            result.update(
                {
                    "target_utilization": target,
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "min_buy": str(sliced["buy_date"].min()) if len(sliced) else "",
                    "max_buy": str(sliced["buy_date"].max()) if len(sliced) else "",
                    "mean_daily_target_sum": float(daily_sum.mean()) if len(daily_sum) else 0.0,
                    "max_daily_target_sum": float(daily_sum.max()) if len(daily_sum) else 0.0,
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        "case": result["case"],
                        "slice": slice_name,
                        "annual": result.get("pnl_ratio_annual"),
                        "sharpe": result.get("sharp_ratio"),
                        "mdd": result.get("max_drawdown"),
                        "returncode": result.get("returncode"),
                        "parse": result.get("parse_status"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 同源信号仓位利用率调参 20260715",
        "",
        "## 结论",
        "",
        "本轮不补位、不扩候选，只把已经通过的 `ogd_gap1_x050` 同源信号按日目标仓位提高到 70%、85%、100%。",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 平均仓位 | 开仓 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(["case", "slice"]).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {int(row.get('open_count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

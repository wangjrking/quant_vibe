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
SIGNAL_DIR = REPORT_DIR / "signals" / "open_quality_boost"
LOG_DIR = REPORT_DIR / "logs" / "open_quality_boost_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_open_quality_boost_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_open_quality_boost_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_open_quality_boost_review_20260715.md"

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
    {"case": "oqb_m3_0_x125", "gap_min": -3.0, "gap_max": 0.0, "scale": 1.25},
    {"case": "oqb_m3_0_x150", "gap_min": -3.0, "gap_max": 0.0, "scale": 1.50},
    {"case": "oqb_m5_p05_x125", "gap_min": -5.0, "gap_max": 0.5, "scale": 1.25},
    {"case": "oqb_m5_p05_x150", "gap_min": -5.0, "gap_max": 0.5, "scale": 1.50},
    {"case": "oqb_m3_p05_x125", "gap_min": -3.0, "gap_max": 0.5, "scale": 1.25},
    {"case": "oqb_m3_p05_x150", "gap_min": -3.0, "gap_max": 0.5, "scale": 1.50},
]
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
    gap = pd.to_numeric(out.get("exec_open_gap_pct", out.get("buy_open_gap_pct")), errors="coerce")
    raw = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    mask = gap.ge(case["gap_min"]) & gap.le(case["gap_max"])
    out["target_pct_before_open_quality_boost"] = raw
    out["open_quality_boosted"] = mask
    out["target_pct"] = raw.mask(mask, raw * float(case["scale"]))
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).where(daily_sum > 1.0, 1.0)
    out["target_pct"] = out["target_pct"] * cap_scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
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
        "stdout_tail": (proc.stdout or "")[-800:],
        "stderr_tail": (proc.stderr or "")[-800:],
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
            path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_juejin(path, case["case"], slice_name)
            result.update(
                {
                    "gap_min": case["gap_min"],
                    "gap_max": case["gap_max"],
                    "scale": case["scale"],
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "boosted_rows": int(sliced.get("open_quality_boosted", pd.Series(dtype=bool)).sum()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
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
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 同源信号买入日开盘质量加仓调参 20260715",
        "",
        "## 口径",
        "",
        "不补位、不扩候选，只对 `ogd_gap1_x050` 中买入日开盘涨幅位于指定区间的原信号加仓，单日总仓位不超过 100%。",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 平均仓位 | 加仓行数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(["case", "slice"]).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {int(row.get('boosted_rows', 0))} |"
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

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
SOURCE = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_four_year_l4_frequency_optimization_20260714"
    / "signals"
    / "latest_core_plus_tiny_refill"
    / "ogd_deep8_x2p0_rf03_empty_full.csv"
)
SIGNAL_DIR = REPORT_DIR / "signals" / "full_refill_scale_smooth"
LOG_DIR = REPORT_DIR / "logs" / "full_refill_scale_smooth_20260715"
OUT_CSV = REPORT_DIR / "full_refill_scale_smooth_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "full_refill_scale_smooth_juejin_20260715.json"
OUT_MD = REPORT_DIR / "full_refill_scale_smooth_review_20260715.md"

RUNNER = MAIN / "run_juejin_signal_backtest.py"
STRATEGY_DIR = MAIN / "strategy_library" / "production" / "prod_fw_soft_deepdrop_weight_v20260706" / "code_snapshot"
SCORE_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l4_executable_10d_open_return_formal.duckdb"
SCORE_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
MARKET_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

FULL_CASES = [
    {"case": "frs_scale035_cap060", "scale": 0.35, "cap": 0.60, "refill_cap": 0.03},
    {"case": "frs_scale040_cap060", "scale": 0.40, "cap": 0.60, "refill_cap": 0.03},
    {"case": "frs_scale045_cap060", "scale": 0.45, "cap": 0.60, "refill_cap": 0.03},
    {"case": "frs_scale050_cap060", "scale": 0.50, "cap": 0.60, "refill_cap": 0.03},
    {"case": "frs_scale055_cap065", "scale": 0.55, "cap": 0.65, "refill_cap": 0.03},
    {"case": "frs_scale060_cap065", "scale": 0.60, "cap": 0.65, "refill_cap": 0.03},
    {"case": "frs_scale065_cap070", "scale": 0.65, "cap": 0.70, "refill_cap": 0.03},
    {"case": "frs_scale070_cap070", "scale": 0.70, "cap": 0.70, "refill_cap": 0.03},
    {"case": "frs_scale075_cap075", "scale": 0.75, "cap": 0.75, "refill_cap": 0.03},
    {"case": "frs_scale080_cap080", "scale": 0.80, "cap": 0.80, "refill_cap": 0.03},
    {"case": "frs_scale090_cap090", "scale": 0.90, "cap": 0.90, "refill_cap": 0.03},
    {"case": "frs_scale100_cap100_ref", "scale": 1.00, "cap": 1.00, "refill_cap": 0.03},
    {"case": "frs_core060_refill015", "scale": 0.60, "cap": 0.65, "refill_cap": 0.015},
    {"case": "frs_core070_refill015", "scale": 0.70, "cap": 0.70, "refill_cap": 0.015},
    {"case": "frs_core080_refill015", "scale": 0.80, "cap": 0.80, "refill_cap": 0.015},
]
SLICE_STARTS = {
    "full": None,
    "from_202407": "20240701",
    "from_202501": "20250101",
    "recent60": "__recent60__",
}


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


def slice_frame(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    start = SLICE_STARTS[slice_name]
    if start == "__recent60__":
        dates = sorted(df["buy_date"].dropna().astype(str).unique())
        return df[df["buy_date"].astype(str).isin(set(dates[-60:]))].copy()
    if start:
        return df[df["buy_date"].astype(str) >= start].copy()
    return df.copy()


def build_case(source: pd.DataFrame, case: dict, slice_name: str) -> pd.DataFrame:
    df = slice_frame(source, slice_name)
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    layer = df.get("layer", pd.Series("", index=df.index)).fillna("").astype(str)
    scaled = target * float(case["scale"])
    refill = layer.eq("tiny_refill")
    scaled.loc[refill] = target.loc[refill].clip(upper=float(case["refill_cap"]))
    df["target_pct"] = scaled.clip(lower=0.0, upper=float(case["cap"]))
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = f"{case['case']}_{slice_name}"
    df["filter_name"] = f"{case['case']}_{slice_name}"
    df["full_refill_smooth_case"] = case["case"]
    df["rank"] = df.groupby("buy_date")["target_pct"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["buy_date", "rank"]).copy()


def run_juejin(signal_file: Path, log_file: Path, max_positions: int) -> dict:
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
        "1",
        "--max-holding-days",
        "3",
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
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows: list[dict] = []
    # First run all cases on full. Then run all slices only for full target hits.
    for case in FULL_CASES:
        slices = ["full"]
        if any(row.get("case") == case["case"] and row.get("slice") == "full" for row in rows):
            slices = []
        for slice_name in slices:
            signal = build_case(source, case, slice_name)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            signal.to_csv(out_path, index=False, encoding="utf-8-sig")
            max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
            log_file = LOG_DIR / f"{case['case']}_{slice_name}_mp{max_positions}.log"
            result = run_juejin(out_path, log_file, max_positions)
            daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
            result.update(
                {
                    **case,
                    "slice": slice_name,
                    "signal_file": str(out_path),
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
            print(json.dumps({"case": case["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    full_hits = frame[
        frame["slice"].eq("full")
        & (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False])
    selected_cases = list(full_hits["case"].head(5))
    for case in FULL_CASES:
        if case["case"] not in selected_cases:
            continue
        for slice_name in ["from_202407", "from_202501", "recent60"]:
            signal = build_case(source, case, slice_name)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            signal.to_csv(out_path, index=False, encoding="utf-8-sig")
            max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
            log_file = LOG_DIR / f"{case['case']}_{slice_name}_mp{max_positions}.log"
            result = run_juejin(out_path, log_file, max_positions)
            daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
            result.update(
                {
                    **case,
                    "slice": slice_name,
                    "signal_file": str(out_path),
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
            print(json.dumps({"case": case["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    full = frame[frame["slice"].eq("full")].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 完整高覆盖信号仓位平滑验证 20260715",
        "",
        "## 口径",
        "",
        "- 输入：`ogd_deep8_x2p0_rf03_empty_full` 完整信号，覆盖到 20260714。",
        "- 不改选股来源，只调仓位缩放、单票上限和 tiny refill 上限。",
        "- 目标：提高信号通过比例，同时把 full Sharpe 拉到 4 以上。",
        "",
        "## full 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in full.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {int(row.get('buy_days') or 0)} | {int(row.get('open_count') or 0)} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    if selected_cases:
        lines.extend(["", "## full 达标候选切片复核", "", "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 买入日 | 最新买入日 |", "|---|---|---:|---:|---:|---:|---|"])
        for row in frame[frame["case"].isin(selected_cases)].sort_values(["case", "slice"]).to_dict("records"):
            lines.append(
                f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
                f"{float(row.get('sharp_ratio') or 0):.4f} | {pct(row.get('max_drawdown'))} | "
                f"{int(row.get('buy_days') or 0)} | {row.get('max_buy_date')} |"
            )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

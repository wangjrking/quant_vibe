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
SIGNAL_DIR = REPORT_DIR / "signals" / "buy_quality_filter"
LOG_DIR = REPORT_DIR / "logs" / "buy_quality_filter_20260715"
OUT_CSV = REPORT_DIR / "pass_ratio_buy_quality_filter_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pass_ratio_buy_quality_filter_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pass_ratio_buy_quality_filter_review_20260715.md"

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

SLICES = [
    ("full", None),
    ("from_202501", "20250102"),
    ("recent60", "__recent60__"),
]

CASES = [
    {"case": "bq_shallow3_x050", "rules": [("pct_gt", -3.0, 0.50)]},
    {"case": "bq_shallow3_x025", "rules": [("pct_gt", -3.0, 0.25)]},
    {"case": "bq_shallow4_x050", "rules": [("pct_gt", -4.0, 0.50)]},
    {"case": "bq_lowamt20_x050", "rules": [("amount_lt", 200000.0, 0.50)]},
    {"case": "bq_lowturn2_x050", "rules": [("turn_lt", 2.0, 0.50)]},
    {"case": "bq_highp5_x050", "rules": [("pred5_ge", 0.9980, 0.50)]},
    {"case": "bq_highp10_x050", "rules": [("pred10_ge", 0.9980, 0.50)]},
    {
        "case": "bq_combo_soft",
        "rules": [("pct_gt", -3.0, 0.65), ("amount_lt", 200000.0, 0.70), ("pred5_ge", 0.9980, 0.70)],
    },
    {
        "case": "bq_combo_mid",
        "rules": [("pct_gt", -3.0, 0.50), ("amount_lt", 200000.0, 0.60), ("pred5_ge", 0.9980, 0.60)],
    },
    {
        "case": "bq_combo_hard",
        "rules": [("pct_gt", -3.0, 0.35), ("amount_lt", 200000.0, 0.50), ("pred5_ge", 0.9980, 0.50)],
    },
]


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


def as_num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(float("nan"), index=frame.index)


def apply_rules(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    out = source.copy()
    scale = pd.Series(1.0, index=out.index, dtype=float)
    hit_any = pd.Series(False, index=out.index)
    for kind, threshold, factor in case["rules"]:
        if kind == "pct_gt":
            mask = as_num(out, "signal_pct_chg_raw") > float(threshold)
        elif kind == "amount_lt":
            mask = as_num(out, "amount") < float(threshold)
        elif kind == "turn_lt":
            mask = as_num(out, "turnover_rate") < float(threshold)
        elif kind == "pred5_ge":
            mask = as_num(out, "pred_5d") >= float(threshold)
        elif kind == "pred10_ge":
            mask = as_num(out, "pred_10d") >= float(threshold)
        else:
            raise ValueError(kind)
        scale = scale.mask(mask, scale * float(factor))
        hit_any = hit_any | mask.fillna(False)
    out["target_pct_before_buy_quality"] = pd.to_numeric(out["target_pct"], errors="coerce").fillna(0.0)
    out["buy_quality_scaled"] = hit_any
    out["buy_quality_scale"] = scale
    out["target_pct"] = out["target_pct_before_buy_quality"] * scale
    out = out[out["target_pct"] > 0].copy()
    daily_sum = out.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).where(daily_sum > 1.0, 1.0)
    out["target_pct"] = out["target_pct"] * cap_scale
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    return out


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


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
        built = apply_rules(source, case)
        for slice_name, start in SLICES:
            sliced = slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_juejin(out_path, case["case"], slice_name)
            result.update(
                {
                    "rules": json.dumps(case["rules"], ensure_ascii=False),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "scaled_rows": int(sliced.get("buy_quality_scaled", pd.Series(dtype=bool)).sum()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                }
            )
            rows.append(result)
            print(json.dumps({"case": result["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# 同源信号买入质量过滤/降权掘金复跑 20260715",
        "",
        "| 版本 | 切片 | 年化 | Sharpe | 最大回撤 | 平均仓位 | 降权行数 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values(["case", "slice"]).to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {row['slice']} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {int(row.get('scaled_rows', 0))} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

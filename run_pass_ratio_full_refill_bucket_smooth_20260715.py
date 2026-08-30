from __future__ import annotations

import importlib.util
import json
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
SIGNAL_DIR = REPORT_DIR / "signals" / "full_refill_bucket_smooth"
LOG_DIR = REPORT_DIR / "logs" / "full_refill_bucket_smooth_20260715"
OUT_CSV = REPORT_DIR / "full_refill_bucket_smooth_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "full_refill_bucket_smooth_juejin_20260715.json"
OUT_MD = REPORT_DIR / "full_refill_bucket_smooth_review_20260715.md"


CASES = [
    {"case": "frb_q2_100_q1_050_q0_020_rf015", "q2": 1.00, "q1": 0.50, "q0": 0.20, "rf": 0.015, "cap": 0.90},
    {"case": "frb_q2_100_q1_060_q0_025_rf015", "q2": 1.00, "q1": 0.60, "q0": 0.25, "rf": 0.015, "cap": 0.90},
    {"case": "frb_q2_095_q1_060_q0_030_rf015", "q2": 0.95, "q1": 0.60, "q0": 0.30, "rf": 0.015, "cap": 0.90},
    {"case": "frb_q2_090_q1_060_q0_035_rf020", "q2": 0.90, "q1": 0.60, "q0": 0.35, "rf": 0.020, "cap": 0.90},
    {"case": "frb_q2_085_q1_065_q0_040_rf020", "q2": 0.85, "q1": 0.65, "q0": 0.40, "rf": 0.020, "cap": 0.90},
    {"case": "frb_q2_100_q1_040_q0_015_rf010", "q2": 1.00, "q1": 0.40, "q0": 0.15, "rf": 0.010, "cap": 0.90},
    {"case": "frb_q2_110_q1_045_q0_015_rf010", "q2": 1.10, "q1": 0.45, "q0": 0.15, "rf": 0.010, "cap": 1.00},
    {"case": "frb_q2_115_q1_050_q0_020_rf010", "q2": 1.15, "q1": 0.50, "q0": 0.20, "rf": 0.010, "cap": 1.00},
    {"case": "frb_q2_120_q1_055_q0_020_rf015", "q2": 1.20, "q1": 0.55, "q0": 0.20, "rf": 0.015, "cap": 1.00},
    {"case": "frb_q2_125_q1_050_q0_015_rf010", "q2": 1.25, "q1": 0.50, "q0": 0.15, "rf": 0.010, "cap": 1.00},
    {"case": "frb_q2_100_q1_000_q0_000_rf020", "q2": 1.00, "q1": 0.00, "q0": 0.00, "rf": 0.020, "cap": 1.00},
    {"case": "frb_q2_100_q1_030_q0_000_rf015", "q2": 1.00, "q1": 0.30, "q0": 0.00, "rf": 0.015, "cap": 1.00},
]


def build_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source.copy()
    bucket = pd.to_numeric(df["quality_bucket"], errors="coerce").fillna(-1).astype(int)
    layer = df.get("layer", pd.Series("", index=df.index)).fillna("").astype(str)
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    scale = pd.Series(1.0, index=df.index)
    scale.loc[bucket == 2] = float(case["q2"])
    scale.loc[bucket == 1] = float(case["q1"])
    scale.loc[bucket == 0] = float(case["q0"])
    scaled = target * scale
    refill = layer.eq("tiny_refill")
    scaled.loc[refill] = target.loc[refill].clip(upper=float(case["rf"]))
    df["target_pct"] = scaled.clip(lower=0.0, upper=float(case["cap"]))
    df = df[df["target_pct"] > 0].copy()
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["full_refill_bucket_smooth_case"] = case["case"]
    df["rank"] = df.groupby("buy_date")["target_pct"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["buy_date", "rank"]).copy()


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(base.SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows = []
    for case in CASES:
        signal = build_case(source, case)
        out_path = SIGNAL_DIR / f"{case['case']}_full.csv"
        signal.to_csv(out_path, index=False, encoding="utf-8-sig")
        max_positions = int(max(signal.groupby("buy_date")["stock_code"].count().max(), 1)) if len(signal) else 1
        log_file = LOG_DIR / f"{case['case']}_full_mp{max_positions}.log"
        result = base.run_juejin(out_path, log_file, max_positions)
        daily = signal.groupby("buy_date")["target_pct"].sum() if len(signal) else pd.Series(dtype=float)
        result.update(
            {
                **case,
                "slice": "full",
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
        print(json.dumps({"case": case["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    full = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 完整高覆盖信号质量桶平滑验证 20260715",
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
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

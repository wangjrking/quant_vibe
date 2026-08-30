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
SIGNAL_DIR = REPORT_DIR / "signals" / "pct_gap_weight_smooth"
LOG_DIR = REPORT_DIR / "logs" / "pct_gap_weight_smooth_20260715"
OUT_CSV = REPORT_DIR / "pct_gap_weight_smooth_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "pct_gap_weight_smooth_juejin_20260715.json"
OUT_MD = REPORT_DIR / "pct_gap_weight_smooth_review_20260715.md"


CASES = [
    {"case": "pgw_d100_m085_w050_s010_gn090_gp070_cap090", "deep": 1.00, "mid": 0.85, "weak": 0.50, "shallow": 0.10, "gap_neg": 0.90, "gap_pos": 0.70, "refill": 0.010, "cap": 0.90},
    {"case": "pgw_d105_m085_w045_s005_gn090_gp060_cap095", "deep": 1.05, "mid": 0.85, "weak": 0.45, "shallow": 0.05, "gap_neg": 0.90, "gap_pos": 0.60, "refill": 0.008, "cap": 0.95},
    {"case": "pgw_d110_m090_w040_s005_gn085_gp055_cap100", "deep": 1.10, "mid": 0.90, "weak": 0.40, "shallow": 0.05, "gap_neg": 0.85, "gap_pos": 0.55, "refill": 0.006, "cap": 1.00},
    {"case": "pgw_d115_m090_w035_s000_gn085_gp050_cap100", "deep": 1.15, "mid": 0.90, "weak": 0.35, "shallow": 0.00, "gap_neg": 0.85, "gap_pos": 0.50, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d120_m095_w030_s000_gn080_gp045_cap100", "deep": 1.20, "mid": 0.95, "weak": 0.30, "shallow": 0.00, "gap_neg": 0.80, "gap_pos": 0.45, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d125_m095_w025_s000_gn080_gp040_cap100", "deep": 1.25, "mid": 0.95, "weak": 0.25, "shallow": 0.00, "gap_neg": 0.80, "gap_pos": 0.40, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d100_m075_w030_s005_gn100_gp050_cap090", "deep": 1.00, "mid": 0.75, "weak": 0.30, "shallow": 0.05, "gap_neg": 1.00, "gap_pos": 0.50, "refill": 0.006, "cap": 0.90},
    {"case": "pgw_d110_m080_w025_s005_gn100_gp045_cap100", "deep": 1.10, "mid": 0.80, "weak": 0.25, "shallow": 0.05, "gap_neg": 1.00, "gap_pos": 0.45, "refill": 0.006, "cap": 1.00},
    {"case": "pgw_d120_m085_w020_s000_gn095_gp040_cap100", "deep": 1.20, "mid": 0.85, "weak": 0.20, "shallow": 0.00, "gap_neg": 0.95, "gap_pos": 0.40, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d130_m090_w015_s000_gn095_gp035_cap100", "deep": 1.30, "mid": 0.90, "weak": 0.15, "shallow": 0.00, "gap_neg": 0.95, "gap_pos": 0.35, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d140_m095_w010_s000_gn090_gp030_cap100", "deep": 1.40, "mid": 0.95, "weak": 0.10, "shallow": 0.00, "gap_neg": 0.90, "gap_pos": 0.30, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d150_m100_w010_s000_gn090_gp025_cap100", "deep": 1.50, "mid": 1.00, "weak": 0.10, "shallow": 0.00, "gap_neg": 0.90, "gap_pos": 0.25, "refill": 0.000, "cap": 1.00},
    {"case": "pgw_d120_m100_w070_s020_gn100_gp100_cap100", "deep": 1.20, "mid": 1.00, "weak": 0.70, "shallow": 0.20, "gap_neg": 1.00, "gap_pos": 1.00, "refill": 0.020, "cap": 1.00},
    {"case": "pgw_d140_m115_w065_s015_gn100_gp100_cap100", "deep": 1.40, "mid": 1.15, "weak": 0.65, "shallow": 0.15, "gap_neg": 1.00, "gap_pos": 1.00, "refill": 0.015, "cap": 1.00},
    {"case": "pgw_d160_m125_w055_s010_gn100_gp095_cap100", "deep": 1.60, "mid": 1.25, "weak": 0.55, "shallow": 0.10, "gap_neg": 1.00, "gap_pos": 0.95, "refill": 0.010, "cap": 1.00},
    {"case": "pgw_d180_m140_w045_s005_gn100_gp090_cap100", "deep": 1.80, "mid": 1.40, "weak": 0.45, "shallow": 0.05, "gap_neg": 1.00, "gap_pos": 0.90, "refill": 0.008, "cap": 1.00},
    {"case": "pgw_d200_m150_w040_s005_gn100_gp085_cap100", "deep": 2.00, "mid": 1.50, "weak": 0.40, "shallow": 0.05, "gap_neg": 1.00, "gap_pos": 0.85, "refill": 0.006, "cap": 1.00},
    {"case": "pgw_d220_m160_w035_s000_gn100_gp080_cap100", "deep": 2.20, "mid": 1.60, "weak": 0.35, "shallow": 0.00, "gap_neg": 1.00, "gap_pos": 0.80, "refill": 0.000, "cap": 1.00},
]


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


def build_case(source: pd.DataFrame, case: dict) -> pd.DataFrame:
    df = source.copy()
    target = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    signal_pct = pd.to_numeric(df["signal_pct_chg_raw"], errors="coerce")
    gap = pd.to_numeric(df.get("exec_open_gap_pct", df.get("buy_open_gap_raw_pct")), errors="coerce")
    layer = df.get("layer", pd.Series("", index=df.index)).fillna("").astype(str)

    scale = pd.Series(float(case["weak"]), index=df.index)
    scale.loc[signal_pct <= -6.0] = float(case["deep"])
    scale.loc[(signal_pct > -6.0) & (signal_pct <= -4.0)] = float(case["mid"])
    scale.loc[(signal_pct > -2.0) | signal_pct.isna()] = float(case["shallow"])

    # Execution-time open gap is a hard observable at the buy open. Downweight
    # ordinary/high opens that historically diluted Sharpe, without dropping
    # the signal row from coverage.
    gap_scale = pd.Series(1.0, index=df.index)
    gap_scale.loc[(gap > -2.0) & (gap <= 1.5)] = float(case["gap_pos"])
    gap_scale.loc[gap <= -2.0] = float(case["gap_neg"])
    gap_scale.loc[gap > 1.5] = 0.0

    scaled = target * scale * gap_scale
    refill = layer.eq("tiny_refill")
    scaled.loc[refill] = scaled.loc[refill].clip(upper=float(case["refill"]))
    df["target_pct"] = scaled.clip(lower=0.0, upper=float(case["cap"]))
    df = df[df["target_pct"] > 0].copy()
    if df.empty:
        return df
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    cap_scale = (1.0 / daily_sum).clip(upper=1.0)
    df["target_pct"] = df["target_pct"] * cap_scale
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    df["strategy_variant"] = case["case"]
    df["filter_name"] = case["case"]
    df["pct_gap_weight_smooth_case"] = case["case"]
    df["rank"] = df.groupby("buy_date")["target_pct"].rank(method="first", ascending=False).astype(int)
    return df.sort_values(["buy_date", "rank"]).copy()


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
        print(json.dumps({"case": case["case"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "buy_days": result["buy_days"]}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 信号日跌幅与买入开盘缺口分层仓位验证 20260715",
        "",
        "## 口径",
        "",
        "- 输入：当前 active L4 生产资产重建的高覆盖 full 信号 `ogd_deep8_x2p0_rf03_empty_full`。",
        "- 不新增候补票；保留原信号，通过信号日跌幅和买入日不复权开盘缺口调节仓位。",
        "- 买入日开盘缺口只作为执行时可观察门控/调仓变量，高开超过 1.5% 的行降为 0。",
        "- 本轮为 research-only，指标以掘金日志为准。",
        "",
        "## full 结果",
        "",
        "| 版本 | 年化 | Sharpe | 最大回撤 | 买入日 | 开仓 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {pct(row.get('pnl_ratio_annual'))} | {num(row.get('sharp_ratio')):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {integer(row.get('buy_days'))} | {integer(row.get('open_count'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 结果 CSV：`{OUT_CSV}`", f"- 结果 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

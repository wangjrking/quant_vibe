from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("juejin_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
CORE_SIGNAL = REPORT_DIR / "signals" / "sell_frequency_fine" / "sf_base_full.csv"
DAILY_TOP1_RESULTS = REPORT_DIR / "daily_top1_pass_lift_juejin_20260715.csv"
DAILY_TOP1_SIGNAL_DIR = REPORT_DIR / "signals" / "daily_top1_pass_lift"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_daily_top1_overlay"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_daily_top1_overlay_20260715"
OUT_CSV = REPORT_DIR / "core_plus_daily_top1_overlay_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "core_plus_daily_top1_overlay_juejin_20260715.json"
OUT_MD = REPORT_DIR / "core_plus_daily_top1_overlay_review_20260715.md"


OVERLAY_SOURCES = [
    "dt1_lift_p150pos_s035",
    "dt1_all_tiny_s020",
    "dt1_all_rankguard_s030",
]
CASES = [
    {"case": "core_only_ref", "overlay": None, "overlay_scale": 0.0, "overlay_cap": 0.0},
    {"case": "core_plus_p150pos_x025", "overlay": "dt1_lift_p150pos_s035", "overlay_scale": 0.25, "overlay_cap": 0.05},
    {"case": "core_plus_p150pos_x050", "overlay": "dt1_lift_p150pos_s035", "overlay_scale": 0.50, "overlay_cap": 0.08},
    {"case": "core_plus_alltiny_x025", "overlay": "dt1_all_tiny_s020", "overlay_scale": 0.25, "overlay_cap": 0.04},
    {"case": "core_plus_alltiny_x050", "overlay": "dt1_all_tiny_s020", "overlay_scale": 0.50, "overlay_cap": 0.06},
    {"case": "core_plus_rankguard_x025", "overlay": "dt1_all_rankguard_s030", "overlay_scale": 0.25, "overlay_cap": 0.05},
    {"case": "core_plus_rankguard_x050", "overlay": "dt1_all_rankguard_s030", "overlay_scale": 0.50, "overlay_cap": 0.08},
]


def load_overlay(name: str) -> pd.DataFrame:
    path = DAILY_TOP1_SIGNAL_DIR / f"{name}_full.csv"
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    return df


def build_case(case: dict) -> pd.DataFrame:
    core = pd.read_csv(CORE_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    core = core.copy()
    core["overlay_source"] = "core"
    if not case["overlay"]:
        out = core
    else:
        overlay = load_overlay(case["overlay"])
        core_days = set(core["buy_date"].astype(str))
        overlay = overlay[~overlay["buy_date"].astype(str).isin(core_days)].copy()
        overlay["target_pct"] = (
            pd.to_numeric(overlay["target_pct"], errors="coerce").fillna(0.0)
            * float(case["overlay_scale"])
        ).clip(upper=float(case["overlay_cap"]))
        overlay = overlay[overlay["target_pct"] > 0].copy()
        overlay["overlay_source"] = case["overlay"]
        out = pd.concat([core, overlay], ignore_index=True, sort=False)
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["source_strategy_variant"] = "core_signal_plus_daily_top1_overlay_no_refill"
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out = out.sort_values(["buy_date", "overlay_source", "target_pct"], ascending=[True, True, False])
    return out


def run_juejin(signal_file: Path, case: str, slice_name: str) -> dict:
    old_log_dir = base.LOG_DIR
    base.LOG_DIR = LOG_DIR
    try:
        return base.run_juejin(signal_file, case, slice_name)
    finally:
        base.LOG_DIR = old_log_dir


def pct(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    daily_meta = pd.read_csv(DAILY_TOP1_RESULTS, encoding="utf-8-sig")
    rows = []
    for case in CASES:
        built = build_case(case)
        for slice_name, start in base.SLICES:
            sliced = base.slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            overlay_rows = int((sliced.get("overlay_source", pd.Series(dtype=str)).astype(str) != "core").sum()) if len(sliced) else 0
            result = run_juejin(out_path, case["case"], slice_name)
            result.update(
                {
                    **case,
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "stock_count": int(sliced["stock_code"].nunique()) if len(sliced) else 0,
                    "overlay_rows": overlay_rows,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                    "max_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                }
            )
            rows.append(result)
            print(
                json.dumps(
                    {
                        "case": result["case"],
                        "slice": slice_name,
                        "buy_days": result["buy_days"],
                        "overlay_rows": overlay_rows,
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

    full = frame[frame["slice"].eq("full")].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False])
    lines = [
        "# 核心信号叠加每日 Top1 小仓位通过验证 20260715",
        "",
        "## 口径",
        "",
        "- 不做候补池：核心信号日沿用核心信号；核心无信号日只允许当日主排序 Top1 小仓位通过。",
        "- 目标：提高有信号日期比例，同时尽量不破坏核心高收益。",
        "- 本轮为 research-only，正式指标以掘金日志为准。",
        "",
        "## full 结果",
        "",
        "| 版本 | 买入日 | overlay行 | 年化 | Sharpe | 最大回撤 | 胜率 | 均值仓位 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in full.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {int(row['buy_days'])} | {int(row['overlay_rows'])} | "
            f"{pct(row.get('pnl_ratio_annual'))} | {float(row.get('sharp_ratio') or 0):.4f} | "
            f"{pct(row.get('max_drawdown'))} | {pct(row.get('win_ratio'))} | "
            f"{pct(row.get('mean_daily_target_sum'))} | {row.get('max_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 每日 Top1 结果来源：`{DAILY_TOP1_RESULTS}`",
            f"- 结果 CSV：`{OUT_CSV}`",
            f"- 结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

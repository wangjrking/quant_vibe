from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("scale_fine", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

OUT_CSV = base.REPORT_DIR / "pass_ratio_open_verified_scale_extended_juejin_20260715.csv"
OUT_JSON = base.REPORT_DIR / "pass_ratio_open_verified_scale_extended_juejin_20260715.json"
OUT_MD = base.REPORT_DIR / "pass_ratio_open_verified_scale_extended_review_20260715.md"
SIGNAL_DIR = base.REPORT_DIR / "signals" / "open_verified_scale_extended"
LOG_DIR = base.REPORT_DIR / "logs" / "open_verified_scale_extended_20260715"

CASES = [
    {"case": "ov_top1_s135_cap065", "scale": 1.35, "cap": 0.65},
    {"case": "ov_top1_s145_cap065", "scale": 1.45, "cap": 0.65},
    {"case": "ov_top1_s155_cap065", "scale": 1.55, "cap": 0.65},
    {"case": "ov_top1_s165_cap065", "scale": 1.65, "cap": 0.65},
    {"case": "ov_top1_s135_cap070", "scale": 1.35, "cap": 0.70},
    {"case": "ov_top1_s150_cap070", "scale": 1.50, "cap": 0.70},
    {"case": "ov_top1_s165_cap070", "scale": 1.65, "cap": 0.70},
    {"case": "ov_top1_s180_cap070", "scale": 1.80, "cap": 0.70},
]


def run_case(signal_file: Path, case: str, slice_name: str) -> dict:
    old_log_dir = base.LOG_DIR
    base.LOG_DIR = LOG_DIR
    try:
        return base.run_juejin(signal_file, case, slice_name)
    finally:
        base.LOG_DIR = old_log_dir


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(base.SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    rows: list[dict] = []
    for case in CASES:
        built = base.build_case(source, case)
        for slice_name, start in base.SLICES:
            sliced = base.slice_frame(built, start)
            out_path = SIGNAL_DIR / f"{case['case']}_{slice_name}.csv"
            sliced.to_csv(out_path, index=False, encoding="utf-8-sig")
            daily = sliced.groupby("buy_date")["target_pct"].sum() if len(sliced) else pd.Series(dtype=float)
            result = run_case(out_path, case["case"], slice_name)
            result.update(
                {
                    "scale": float(case["scale"]),
                    "cap": float(case["cap"]),
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "mean_daily_target_sum": float(daily.mean()) if len(daily) else 0.0,
                    "max_daily_target_sum": float(daily.max()) if len(daily) else 0.0,
                }
            )
            open_count = result.get("open_count")
            result["row_execution_ratio"] = float(open_count) / float(len(sliced)) if open_count is not None and len(sliced) else None
            rows.append(result)
            print(json.dumps({"case": result["case"], "slice": slice_name, "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    grouped = []
    for case, group in frame.groupby("case"):
        item = {"case": case}
        for sl in ["full", "from_202501", "recent60"]:
            row = group[group["slice"] == sl].iloc[0]
            item[f"{sl}_annual"] = row.get("pnl_ratio_annual")
            item[f"{sl}_sharpe"] = row.get("sharp_ratio")
            item[f"{sl}_mdd"] = row.get("max_drawdown")
        grouped.append(item)
    gframe = pd.DataFrame(grouped)
    gframe["min_annual"] = gframe[["full_annual", "from_202501_annual", "recent60_annual"]].min(axis=1)
    gframe["min_sharpe"] = gframe[["full_sharpe", "from_202501_sharpe", "recent60_sharpe"]].min(axis=1)
    gframe = gframe.sort_values(["min_annual", "min_sharpe"], ascending=False)

    lines = [
        "# 开盘验证 Top1 仓位扩展掘金复跑 20260715",
        "",
        "## 结论",
        "",
        "- 本轮继续不补位、不扩候选，只提高已通过开盘硬门槛 Top1 的目标仓位。",
        "- 用 full、2025以来、recent60 三段同时观察，判断是否仍有平滑准入空间。",
        "",
        "| 版本 | full年化 | full Sharpe | full回撤 | 2025以来年化 | 2025以来Sharpe | recent60年化 | recent60 Sharpe | 三段最小年化 | 三段最小Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gframe.to_dict("records"):
        lines.append(
            f"| `{row['case']}` | {base.pct(row['full_annual'])} | {float(row['full_sharpe']):.4f} | "
            f"{base.pct(row['full_mdd'])} | {base.pct(row['from_202501_annual'])} | "
            f"{float(row['from_202501_sharpe']):.4f} | {base.pct(row['recent60_annual'])} | "
            f"{float(row['recent60_sharpe']):.4f} | {base.pct(row['min_annual'])} | {float(row['min_sharpe']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
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

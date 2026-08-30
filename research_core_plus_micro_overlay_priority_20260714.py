from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_micro_overlay_priority"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_micro_overlay_priority"
OUT_CSV = REPORT_DIR / "core_plus_micro_overlay_priority_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_plus_micro_overlay_priority_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "核心优先微仓补充信号复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CORE_FILES = {
    "gap_bucket": REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_scale_gap_bucket.csv",
    "bigmv_lowturn": REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_drop_bigmv_lowturn.csv",
}
OVERLAY = REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_04404_p10p1_broad_t2_h2.csv"

CASES = [
    {"case": "prio_gb_empty_s005", "core": "gap_bucket", "mode": "empty", "scale": 0.05, "max_overlay": 1},
    {"case": "prio_gb_lowsum_s005", "core": "gap_bucket", "mode": "lowsum", "scale": 0.05, "max_overlay": 1, "low_sum": 0.45},
    {"case": "prio_bl_empty_s005", "core": "bigmv_lowturn", "mode": "empty", "scale": 0.05, "max_overlay": 1},
    {"case": "prio_bl_lowsum_s005", "core": "bigmv_lowturn", "mode": "lowsum", "scale": 0.05, "max_overlay": 1, "low_sum": 0.45},
]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    score = df.get("sort_score", df.get("entry_score", df.get("pred_prob")))
    df["sort_score"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return df


def build_case(case: dict) -> dict:
    core = load_signal(CORE_FILES[case["core"]]).copy()
    core["layer"] = "core"
    overlay = load_signal(OVERLAY).copy()
    overlay["layer"] = "overlay"
    core_sum = core.groupby("buy_date")["target_pct"].sum().rename("core_sum")
    overlay = overlay.merge(core_sum, on="buy_date", how="left")
    overlay["core_sum"] = overlay["core_sum"].fillna(0.0)
    if case["mode"] == "empty":
        overlay = overlay[overlay["core_sum"] <= 0.0].copy()
    elif case["mode"] == "lowsum":
        overlay = overlay[overlay["core_sum"] < float(case.get("low_sum", 0.45))].copy()
    else:
        raise ValueError(case["mode"])
    core_keys = set(zip(core["buy_date"], core["stock_code"]))
    overlay = overlay[~overlay.apply(lambda r: (r["buy_date"], r["stock_code"]) in core_keys, axis=1)].copy()
    overlay = (
        overlay.sort_values(["buy_date", "sort_score"], ascending=[True, False])
        .groupby("buy_date", group_keys=False)
        .head(int(case["max_overlay"]))
        .copy()
    )
    overlay["target_pct"] = overlay["target_pct"] * float(case["scale"])
    overlay["holding_days"] = 1
    overlay["max_holding_days"] = 3
    overlay["score_exit_entry_ratio"] = "0.98000"
    overlay["score_continue_entry_ratio"] = "1.02000"
    overlay["min_holding_days_before_score_exit"] = 1
    overlay["source_strategy_variant"] = overlay.get("source_strategy_variant", "").astype(str) + "_priority_micro_overlay"

    core = core.sort_values(["buy_date", "rank", "sort_score"], ascending=[True, True, False]).copy()
    core["rank"] = core.groupby("buy_date").cumcount() + 1
    core_count = core.groupby("buy_date")["stock_code"].count().rename("core_count")
    overlay = overlay.merge(core_count, on="buy_date", how="left")
    overlay["core_count"] = overlay["core_count"].fillna(0).astype(int)
    overlay = overlay.sort_values(["buy_date", "sort_score"], ascending=[True, False]).copy()
    overlay["rank"] = overlay["core_count"] + overlay.groupby("buy_date").cumcount() + 1
    overlay = overlay.drop(columns=["core_count"], errors="ignore")

    combined = pd.concat([core, overlay], ignore_index=True, sort=False)
    combined = combined.sort_values(["buy_date", "rank"]).copy()
    combined["strategy_variant"] = case["case"]
    combined["filter_name"] = case["case"]
    combined["priority_micro_overlay_case"] = case["case"]
    combined["daily_target_sum_after_cap"] = combined.groupby("buy_date")["target_pct"].transform("sum")
    out = SIGNAL_DIR / f"{case['case']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    per_day_count = combined.groupby("buy_date")["stock_code"].count()
    return {
        "case": case["case"],
        "signal_file": str(out),
        "core": case["core"],
        "mode": case["mode"],
        "scale": float(case["scale"]),
        "rows": int(len(combined)),
        "core_rows": int(len(core)),
        "overlay_rows": int(len(overlay)),
        "buy_days": int(combined["buy_date"].nunique()),
        "overlay_buy_days": int(overlay["buy_date"].nunique()),
        "stock_count": int(combined["stock_code"].nunique()),
        "max_positions": int(max(per_day_count.max(), 1)),
        "holding_days": 1,
        "max_holding_days": 3,
        "mean_daily_target_sum": float(combined.groupby("buy_date")["target_pct"].sum().mean()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(ind)
            return out
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(base_mod.RUNNER),
        "--strategy-dir",
        str(base_mod.STRATEGY_DIR),
        "--signal-file",
        row["signal_file"],
        "--log-file",
        str(log),
        "--score-db",
        str(base_mod.SCORE_DB),
        "--score-table",
        base_mod.SCORE_TABLE,
        "--market-db",
        str(base_mod.MARKET_DB),
        "--max-positions",
        str(row["max_positions"]),
        "--holding-days",
        "1",
        "--max-holding-days",
        "3",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    ind = base_mod.extract_indicator(text)
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(ind, dict):
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def write_report(frame: pd.DataFrame) -> None:
    lines = [
        "# 核心优先微仓补充信号复核",
        "",
        "## 结论",
        "",
        "- 本轮强制核心信号 rank 优先，补充信号只排在核心之后，避免补充信号污染核心买入顺序。",
        "- 是否准入仍需结合年度切片和去贡献复核。",
        "",
        "## 掘金结果",
        "",
        "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 补充行 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=[False, False]).iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(float(row.get('open_count', 0)))} | {int(float(row.get('buy_days', 0)))} | {int(float(row.get('overlay_rows', 0)))} |"
        )
    lines.extend(["", "## 证据路径", "", f"- 汇总 CSV：`{OUT_CSV}`", f"- 汇总 JSON：`{OUT_JSON}`", f"- 信号目录：`{SIGNAL_DIR}`", f"- 掘金日志目录：`{LOG_DIR}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in CASES:
        row = build_case(case)
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    frame.sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=[False, False]).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

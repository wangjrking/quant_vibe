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
SIGNAL_DIR = REPORT_DIR / "signals" / "core_plus_micro_overlay"
LOG_DIR = REPORT_DIR / "logs" / "core_plus_micro_overlay"
OUT_CSV = REPORT_DIR / "core_plus_micro_overlay_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_plus_micro_overlay_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "核心叠加微仓补充信号复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CORE_FILES = {
    "x125_gap_bucket": REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_scale_gap_bucket.csv",
    "x125_bigmv_lowturn": REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_drop_bigmv_lowturn.csv",
    "x125_shallow2": REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_drop_shallow2.csv",
}

OVERLAY_FILES = {
    "iap044": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_04404_p10p1_broad_t2_h2.csv",
    "iap028": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_02868_p10p1_broad_t1_h2.csv",
}


CASES = [
    {"case": "gb_empty_o044_s002", "core": "x125_gap_bucket", "overlay": ["iap044"], "mode": "empty", "scale": 0.02, "max_overlay": 1},
    {"case": "gb_empty_o044_s005", "core": "x125_gap_bucket", "overlay": ["iap044"], "mode": "empty", "scale": 0.05, "max_overlay": 1},
    {"case": "gb_lowsum_o044_s002", "core": "x125_gap_bucket", "overlay": ["iap044"], "mode": "lowsum", "scale": 0.02, "max_overlay": 1, "low_sum": 0.45},
    {"case": "gb_lowsum_o044_s005", "core": "x125_gap_bucket", "overlay": ["iap044"], "mode": "lowsum", "scale": 0.05, "max_overlay": 1, "low_sum": 0.45},
    {"case": "bl_empty_o044_s002", "core": "x125_bigmv_lowturn", "overlay": ["iap044"], "mode": "empty", "scale": 0.02, "max_overlay": 1},
    {"case": "bl_empty_o044_s005", "core": "x125_bigmv_lowturn", "overlay": ["iap044"], "mode": "empty", "scale": 0.05, "max_overlay": 1},
    {"case": "bl_lowsum_o044_s002", "core": "x125_bigmv_lowturn", "overlay": ["iap044"], "mode": "lowsum", "scale": 0.02, "max_overlay": 1, "low_sum": 0.45},
    {"case": "bl_lowsum_o044_s005", "core": "x125_bigmv_lowturn", "overlay": ["iap044"], "mode": "lowsum", "scale": 0.05, "max_overlay": 1, "low_sum": 0.45},
    {"case": "sh_empty_o028_s003", "core": "x125_shallow2", "overlay": ["iap028"], "mode": "empty", "scale": 0.03, "max_overlay": 1},
    {"case": "sh_lowsum_o028_s003", "core": "x125_shallow2", "overlay": ["iap028"], "mode": "lowsum", "scale": 0.03, "max_overlay": 1, "low_sum": 0.45},
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
    overlays = []
    for name in case["overlay"]:
        item = load_signal(OVERLAY_FILES[name]).copy()
        item["layer"] = name
        overlays.append(item)
    overlay = pd.concat(overlays, ignore_index=True)
    core_sum = core.groupby("buy_date")["target_pct"].sum().rename("core_sum")
    overlay = overlay.merge(core_sum, on="buy_date", how="left")
    overlay["core_sum"] = overlay["core_sum"].fillna(0.0)
    if case["mode"] == "empty":
        overlay = overlay[overlay["core_sum"] <= 0.0].copy()
    elif case["mode"] == "lowsum":
        overlay = overlay[overlay["core_sum"] < float(case.get("low_sum", 0.45))].copy()
    else:
        raise ValueError(case["mode"])
    if len(overlay):
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
        overlay["source_strategy_variant"] = overlay.get("source_strategy_variant", "").astype(str) + "_micro_overlay"
    combined = pd.concat([core, overlay], ignore_index=True, sort=False)
    combined["strategy_variant"] = case["case"]
    combined["filter_name"] = case["case"]
    combined["micro_overlay_case"] = case["case"]
    combined["rank"] = combined.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
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
        "overlay_buy_days": int(overlay["buy_date"].nunique()) if len(overlay) else 0,
        "stock_count": int(combined["stock_code"].nunique()),
        "max_positions": int(max(per_day_count.max(), 1)),
        "holding_days": 1,
        "max_holding_days": 3,
        "mean_daily_target_sum": float(combined.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(combined.groupby("buy_date")["target_pct"].sum().max()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
    if log.exists() and log.stat().st_size > 0:
        indicator = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(indicator, dict) and indicator.get("pnl_ratio_annual") is not None:
            out = dict(row)
            out["returncode"] = 0
            out["log_file"] = str(log)
            out.update(indicator)
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
        str(row["holding_days"]),
        "--max-holding-days",
        str(row["max_holding_days"]),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    indicator = base_mod.extract_indicator(text)
    out = dict(row)
    out["returncode"] = proc.returncode
    out["log_file"] = str(log)
    if isinstance(indicator, dict):
        out.update(indicator)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def write_report(frame: pd.DataFrame) -> None:
    hits = frame[
        (pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce") >= 5.0)
        & (pd.to_numeric(frame["sharp_ratio"], errors="coerce") >= 4.0)
        & (pd.to_numeric(frame["max_drawdown"], errors="coerce") <= 0.4)
    ]
    best = frame.sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=[False, False], na_position="last").head(12)
    lines = [
        "# 核心叠加微仓补充信号复核",
        "",
        "## 结论",
        "",
        f"- 本轮测试 {len(frame)} 个微仓补充候选，其中表面达标 {len(hits)} 个。",
        "- 本轮仅使用 active L4 既有候选信号，不引入日期或未来收益过滤。",
        "- 该复核用于判断低仓位补充是否能改善覆盖且不明显稀释收益；是否准入仍需年度切片和去贡献复核。",
        "",
        "## 掘金结果前 12",
        "",
        "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 补充行 | 平均仓位和 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(float(row.get('open_count', 0)))} | {int(float(row.get('buy_days', 0)))} | "
            f"{int(float(row.get('overlay_rows', 0)))} | {float(row.get('mean_daily_target_sum', 0))*100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 汇总 CSV：`{OUT_CSV}`",
            f"- 汇总 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{SIGNAL_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifests = [build_case(case) for case in CASES]
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        frame = pd.DataFrame(results)
        frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open": result.get("open_count"),
                    "overlay_rows": result.get("overlay_rows"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    frame.sort_values(["pnl_ratio_annual", "sharp_ratio"], ascending=[False, False], na_position="last").to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig"
    )
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

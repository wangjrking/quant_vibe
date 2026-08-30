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
CANDIDATE_CSV = REPORT_DIR / "latest_full_hit_candidates_for_slice_20260714.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "latest_full_hit_slice_admission"
LOG_DIR = REPORT_DIR / "logs" / "latest_full_hit_slice_admission"
OUT_CSV = REPORT_DIR / "latest_full_hit_slice_admission_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "latest_full_hit_slice_admission_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "最新覆盖全周期达标候选切片准入复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


STATIC_SLICES = {
    "slice_2024": ("20240101", "20241231"),
    "slice_2025": ("20250101", "20251231"),
    "slice_2026": ("20260101", "20261231"),
    "slice_2025_2026": ("20250101", "20261231"),
    "slice_from_202407": ("20240701", "20991231"),
    "slice_from_202501": ("20250101", "20991231"),
}


def load_signal(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})


def top_candidates(limit: int = 6) -> pd.DataFrame:
    df = pd.read_csv(CANDIDATE_CSV, encoding="utf-8-sig")
    df["annual"] = pd.to_numeric(df["annual"], errors="coerce")
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")
    df["mdd"] = pd.to_numeric(df["mdd"], errors="coerce")
    df = df.sort_values(["annual", "sharpe"], ascending=False)
    return df.drop_duplicates("signal_file").head(limit).copy()


def make_slice(case: str, signal_file: Path, slice_name: str, start: str, end: str) -> dict | None:
    df = load_signal(signal_file)
    if slice_name == "recent120":
        days = sorted(df["buy_date"].dropna().unique())[-120:]
        sliced = df[df["buy_date"].isin(days)].copy()
    elif slice_name == "recent60":
        days = sorted(df["buy_date"].dropna().unique())[-60:]
        sliced = df[df["buy_date"].isin(days)].copy()
    else:
        sliced = df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()
    if sliced.empty:
        return None
    safe_case = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(case))
    out = SIGNAL_DIR / f"{safe_case}_{slice_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(out, index=False, encoding="utf-8-sig")
    max_positions = int(max(sliced.groupby("buy_date")["stock_code"].count().max(), 1))
    return {
        "case": case,
        "slice": slice_name,
        "signal_file": str(out),
        "source_signal_file": str(signal_file),
        "rows": int(len(sliced)),
        "buy_days": int(sliced["buy_date"].nunique()),
        "stock_count": int(sliced["stock_code"].nunique()),
        "min_buy": str(sliced["buy_date"].min()),
        "max_buy": str(sliced["buy_date"].max()),
        "max_positions": max_positions,
        "holding_days": 1,
        "max_holding_days": 3,
        "mean_daily_target_sum": float(pd.to_numeric(sliced["target_pct"], errors="coerce").groupby(sliced["buy_date"]).sum().mean())
        if "target_pct" in sliced.columns
        else 0.0,
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{Path(row['signal_file']).stem}_mp{row['max_positions']}.log"
    base = dict(row)
    base["log_file"] = str(log)
    if log.exists() and log.stat().st_size > 0:
        ind = base_mod.extract_indicator(log.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(ind, dict) and ind.get("pnl_ratio_annual") is not None:
            out = dict(base)
            out["returncode"] = 0
            out.update(ind)
            return out
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
    out = dict(base)
    out["returncode"] = proc.returncode
    if isinstance(ind, dict):
        out.update(ind)
    else:
        out["indicator_error"] = "missing"
        out["stdout"] = proc.stdout[-1000:]
        out["stderr"] = proc.stderr[-1000:]
    return out


def write_report(frame: pd.DataFrame) -> None:
    def n(value: object, default: float = 0.0) -> float:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            return default
        return float(value)

    full_pass = []
    for case, group in frame.groupby("case"):
        from_202501 = group[group["slice"] == "slice_from_202501"]
        from_202407 = group[group["slice"] == "slice_from_202407"]
        if not from_202501.empty and not from_202407.empty:
            r501 = from_202501.iloc[0]
            r407 = from_202407.iloc[0]
            full_pass.append(
                {
                    "case": case,
                    "from_202501_annual": n(r501.get("pnl_ratio_annual")),
                    "from_202501_sharpe": n(r501.get("sharp_ratio")),
                    "from_202407_annual": n(r407.get("pnl_ratio_annual")),
                    "from_202407_sharpe": n(r407.get("sharp_ratio")),
                }
            )
    pass_df = pd.DataFrame(full_pass)

    lines = [
        "# 最新覆盖全周期达标候选切片准入复核",
        "",
        "## 结论",
        "",
        "- 本报告只复核 `buy_date=20260714` 仍覆盖的全周期表面达标候选。",
        "- 目标是验证这些候选是否在 2025、2026、近期启动下仍满足年化 500%、Sharpe 4、回撤 40% 以下。",
        "- 本报告不发布生产策略，不生成正式交易信号。",
        "",
    ]
    if pass_df.empty:
        lines.append("- 当前没有候选完成完整切片复核。")
    else:
        best_501 = pass_df.sort_values(["from_202501_annual", "from_202501_sharpe"], ascending=False).head(5)
        lines.append("## 2025 起始表现最好的候选")
        lines.append("")
        lines.append("| 候选 | from_202501 年化 | from_202501 Sharpe | from_202407 年化 | from_202407 Sharpe |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in best_501.iterrows():
            lines.append(
                f"| {row['case']} | {row['from_202501_annual'] * 100:.2f}% | "
                f"{row['from_202501_sharpe']:.3f} | {row['from_202407_annual'] * 100:.2f}% | "
                f"{row['from_202407_sharpe']:.3f} |"
            )
        hit = pass_df[
            (pass_df["from_202501_annual"] >= 5.0)
            & (pass_df["from_202501_sharpe"] >= 4.0)
            & (pass_df["from_202407_annual"] >= 5.0)
            & (pass_df["from_202407_sharpe"] >= 4.0)
        ]
        lines.append("")
        if len(hit):
            lines.append(f"- 有 {len(hit)} 个候选通过 from_202501/from_202407 双切片硬指标。")
        else:
            lines.append("- 没有候选通过 from_202501/from_202407 双切片硬指标。")
    lines.extend(
        [
            "",
            "## 全部切片结果",
            "",
            "| 候选 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 行数 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = frame.sort_values(["case", "slice"])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('slice')} | {n(row.get('pnl_ratio_annual')) * 100:.2f}% | "
            f"{n(row.get('sharp_ratio')):.3f} | {n(row.get('max_drawdown')) * 100:.2f}% | "
            f"{int(n(row.get('open_count')))} | {int(n(row.get('buy_days')))} | {int(n(row.get('rows')))} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- 候选清单：`{CANDIDATE_CSV}`",
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
    manifests: list[dict] = []
    slices = dict(STATIC_SLICES)
    slices["recent120"] = ("", "")
    slices["recent60"] = ("", "")
    for _, cand in top_candidates().iterrows():
        src = Path(str(cand["signal_file"]))
        for slice_name, (start, end) in slices.items():
            item = make_slice(str(cand["case"]), src, slice_name, start, end)
            if item:
                manifests.append(item)
    results: list[dict] = []
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
                    "slice": result.get("slice"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "open": result.get("open_count"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

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
SIGNAL_DIR = REPORT_DIR / "signals" / "ogd_conditional_reweight"
LOG_DIR = REPORT_DIR / "logs" / "ogd_conditional_reweight"
OUT_CSV = REPORT_DIR / "ogd_conditional_reweight_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "ogd_conditional_reweight_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "ogd_conditional_reweight_review_20260714.md"

spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)

SOURCES = {
    "ogd": REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_deep8_up110.csv",
    "cpo": REPORT_DIR / "signals" / "core_plus_pullback_overlay" / "cpo_low60_o06_cap100.csv",
}
SLICES = {
    "full": ("00000000", "99999999"),
    "from_202407": ("20240701", "20991231"),
    "from_202501": ("20250101", "20991231"),
    "recent60": ("RECENT60", "RECENT60"),
}
CASES = [
    {"name": "mildhi15_deep25_cap90", "base": 0.85, "mild": 1.50, "deep": 0.25, "cap": 0.90},
    {"name": "mildhi20_deep00_cap90", "base": 0.75, "mild": 2.00, "deep": 0.00, "cap": 0.90},
    {"name": "mildhi20_deep25_cap100", "base": 0.85, "mild": 2.00, "deep": 0.25, "cap": 1.00},
    {"name": "pred1tier_cap90", "base": 0.70, "mild": 1.35, "deep": 0.35, "cap": 0.90, "pred_tier": True},
]


def n(value: object, default: float = 0.0) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(out):
        return default
    return float(out)


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "signal_pct_chg_raw", "pred_1d", "pred_10d", "exec_open_gap_pct", "buy_open_gap_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "exec_open_gap_pct" not in df.columns:
        df["exec_open_gap_pct"] = pd.to_numeric(
            df.get("buy_open_gap_raw_pct", df.get("buy_open_gap_pct")), errors="coerce"
        )
    return df


def apply_rule(df: pd.DataFrame, case: dict) -> pd.Series:
    pct = df["signal_pct_chg_raw"]
    p1 = df["pred_1d"]
    p10 = df["pred_10d"]
    gap = df["exec_open_gap_pct"]
    scale = pd.Series(float(case["base"]), index=df.index)
    deep_bad = pct <= -9.0
    mild_good = pct.between(-5.0, -1.75, inclusive="both") & (p1 >= 0.90) & (p10 >= 0.99) & (gap <= 1.5)
    scale.loc[deep_bad] = float(case["deep"])
    scale.loc[mild_good] = float(case["mild"])
    if case.get("pred_tier"):
        scale.loc[p1 >= 0.99] *= 1.15
        scale.loc[p1 < 0.50] *= 0.50
    return scale.clip(lower=0.0)


def slice_df(df: pd.DataFrame, slice_name: str, start: str, end: str) -> pd.DataFrame:
    if slice_name == "recent60":
        days = sorted(df["buy_date"].dropna().unique())[-60:]
        return df[df["buy_date"].isin(days)].copy()
    return df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()


def build_signal(source: str, path: Path, case: dict, slice_name: str, start: str, end: str) -> dict | None:
    df = slice_df(load_signal(path), slice_name, start, end)
    if df.empty:
        return None
    raw = df["target_pct"].fillna(0.0) * apply_rule(df, case)
    daily = raw.groupby(df["buy_date"]).transform("sum")
    factor = (float(case["cap"]) / daily).clip(upper=1.0).fillna(0.0)
    df["target_pct"] = (raw * factor).clip(lower=0.0)
    df = df[df["target_pct"] > 0].copy()
    if df.empty:
        return None
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case_name = f"{source}_{case['name']}_{slice_name}"
    df["strategy_variant"] = case_name
    df["filter_name"] = case_name
    out = SIGNAL_DIR / f"{case_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily_sum = df.groupby("buy_date")["target_pct"].sum()
    return {
        "case": case_name,
        "source": source,
        "rule": case["name"],
        "slice": slice_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "min_buy": str(df["buy_date"].min()),
        "max_buy": str(df["buy_date"].max()),
        "max_positions": int(max(df.groupby("buy_date")["stock_code"].count().max(), 1)),
        "mean_daily_target_sum": float(daily_sum.mean()),
        "max_daily_target_sum": float(daily_sum.max()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_mp{row['max_positions']}.log"
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
    lines = [
        "# OGD 条件调仓复核",
        "",
        "## 说明",
        "",
        "- 本轮只使用信号日前可观测字段：`signal_pct_chg_raw`、`pred_1d`、`pred_10d`、`exec_open_gap_pct`。",
        "- 不按日期切换策略；日期切片只用于验证后段稳定性。",
        "- 本轮不修改生产策略，不生成正式信号。",
        "",
        "## 结果",
        "",
        "| 候选 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 平均目标仓位 | 覆盖到 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    sort = frame.sort_values(["slice", "pnl_ratio_annual", "sharp_ratio"], ascending=[True, False, False])
    for _, row in sort.iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('slice')} | {n(row.get('pnl_ratio_annual')) * 100:.2f}% | "
            f"{n(row.get('sharp_ratio')):.3f} | {n(row.get('max_drawdown')) * 100:.2f}% | "
            f"{int(n(row.get('open_count')))} | {int(n(row.get('buy_days')))} | "
            f"{n(row.get('mean_daily_target_sum')) * 100:.2f}% | {row.get('max_buy')} |"
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
    manifests: list[dict] = []
    for source, path in SOURCES.items():
        for case in CASES:
            for slice_name, (start, end) in SLICES.items():
                item = build_signal(source, path, case, slice_name, start, end)
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

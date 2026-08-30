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
SIGNAL_DIR = REPORT_DIR / "signals" / "latest_candidate_position_lift"
LOG_DIR = REPORT_DIR / "logs" / "latest_candidate_position_lift"
OUT_CSV = REPORT_DIR / "latest_candidate_position_lift_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "latest_candidate_position_lift_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "最新覆盖候选仓位放大复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "ogd_deep8": REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_deep8_up110.csv",
    "cpo_low60": REPORT_DIR / "signals" / "core_plus_pullback_overlay" / "cpo_low60_o06_cap100.csv",
}

SCALES = [1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
CAPS = [0.90, 1.00]
SLICES = {
    "full": ("00000000", "99999999"),
    "from_202407": ("20240701", "20991231"),
    "from_202501": ("20250101", "20991231"),
    "recent60": ("RECENT60", "RECENT60"),
}


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    return df


def build_signal(source_name: str, source_path: Path, scale: float, cap: float, slice_name: str, start: str, end: str) -> dict | None:
    df = load_signal(source_path)
    if slice_name == "recent60":
        days = sorted(df["buy_date"].dropna().unique())[-60:]
        df = df[df["buy_date"].isin(days)].copy()
    else:
        df = df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()
    if df.empty:
        return None
    raw = df["target_pct"] * float(scale)
    raw_daily = raw.groupby(df["buy_date"]).transform("sum")
    factor = (float(cap) / raw_daily).clip(upper=1.0)
    df["target_pct"] = (raw * factor).clip(lower=0.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = f"{source_name}_x{str(scale).replace('.', 'p')}_cap{int(cap * 100)}_{slice_name}"
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily_sum = df.groupby("buy_date")["target_pct"].sum()
    return {
        "case": case,
        "source": source_name,
        "scale": float(scale),
        "cap": float(cap),
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
    def n(value: object, default: float = 0.0) -> float:
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(value):
            return default
        return float(value)

    lines = [
        "# 最新覆盖候选仓位放大复核",
        "",
        "## 结论",
        "",
        "- 本报告只验证目标仓位倍率和日总仓上限，不使用日期过滤作为策略规则。",
        "- 重点看提高仓位后，`from_202501` 与 `recent60` 是否能接近 500% 年化目标。",
        "- 本报告不发布生产策略，不生成正式交易信号。",
        "",
        "## 最优结果",
        "",
        "| 候选 | 切片 | scale | cap | 年化 | Sharpe | 最大回撤 | 开仓 | 平均日目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    best = frame.sort_values(["slice", "pnl_ratio_annual", "sharp_ratio"], ascending=[True, False, False])
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('slice')} | {n(row.get('scale')):.2f} | {n(row.get('cap')):.2f} | "
            f"{n(row.get('pnl_ratio_annual')) * 100:.2f}% | {n(row.get('sharp_ratio')):.3f} | "
            f"{n(row.get('max_drawdown')) * 100:.2f}% | {int(n(row.get('open_count')))} | "
            f"{n(row.get('mean_daily_target_sum')) * 100:.2f}% |"
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
    for source_name, source_path in SOURCES.items():
        for scale in SCALES:
            for cap in CAPS:
                for slice_name, (start, end) in SLICES.items():
                    item = build_signal(source_name, source_path, scale, cap, slice_name, start, end)
                    if item is not None:
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

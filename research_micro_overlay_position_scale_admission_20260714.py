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
SOURCE_DIR = REPORT_DIR / "signals" / "micro_overlay_quality_boost"
SIGNAL_DIR = REPORT_DIR / "signals" / "micro_overlay_position_scale_admission"
LOG_DIR = REPORT_DIR / "logs" / "micro_overlay_position_scale_admission"
OUT_CSV = REPORT_DIR / "micro_overlay_position_scale_admission_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "micro_overlay_position_scale_admission_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "微仓质量候选仓位放大准入复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SOURCES = {
    "bl_p1hi_sigmid": SOURCE_DIR / "bl_empty_p1hi_sigmid_s015.csv",
    "bl_sigmid": SOURCE_DIR / "bl_empty_sigmid_s012.csv",
}

SCALES = [1.25, 1.50, 1.75, 2.00]
CAPS = [0.75, 0.90, 1.00]
SLICES = {
    "full": ("00000000", "99999999"),
    "from_202501": ("20250101", "99999999"),
    "from_202407": ("20240701", "99999999"),
}


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    return df


def make_scaled(name: str, source: pd.DataFrame, scale: float, cap: float, slice_name: str, start: str, end: str) -> dict | None:
    df = source[(source["buy_date"] >= start) & (source["buy_date"] <= end)].copy()
    if df.empty:
        return None
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    raw = df["target_pct"] * float(scale)
    capped_sum = (daily_sum * float(scale)).clip(upper=float(cap))
    factor = capped_sum / (daily_sum * float(scale)).replace(0.0, pd.NA)
    df["target_pct"] = (raw * factor.fillna(0.0)).clip(lower=0.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    case = f"{name}_x{str(scale).replace('.', 'p')}_cap{int(cap * 100)}_{slice_name}"
    df["strategy_variant"] = case
    df["filter_name"] = case
    out = SIGNAL_DIR / f"{case}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    max_positions = int(max(df.groupby("buy_date")["stock_code"].count().max(), 1))
    return {
        "case": case,
        "source": name,
        "slice": slice_name,
        "scale": scale,
        "cap": cap,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "max_positions": max_positions,
        "holding_days": 1,
        "max_holding_days": 3,
        "mean_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().mean()),
        "max_daily_target_sum": float(df.groupby("buy_date")["target_pct"].sum().max()),
        "min_buy_date": str(df["buy_date"].min()),
        "max_buy_date": str(df["buy_date"].max()),
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
        "1",
        "--max-holding-days",
        "3",
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
    lines = [
        "# 微仓质量候选仓位放大准入复核",
        "",
        "## 结论",
        "",
        "- 本轮固定信号规则，只调整仓位倍率和日目标仓位上限；不使用日期、月份、未来收益过滤。",
        "- 重点观察全周期达标后，2025-2026 独立启动收益是否同步改善。",
        "",
        "## 掘金结果",
        "",
        "| source | slice | scale | cap | 年化 | Sharpe | 最大回撤 | 开仓 | 平均日目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values(["source", "slice", "scale", "cap"]).iterrows():
        lines.append(
            f"| {row.get('source')} | {row.get('slice')} | {float(row.get('scale')):.2f} | "
            f"{float(row.get('cap')):.2f} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(float(row.get('open_count', 0)))} | {float(row.get('mean_daily_target_sum', 0))*100:.2f}% |"
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
    for name, path in SOURCES.items():
        source = load_signal(path)
        for scale in SCALES:
            for cap in CAPS:
                for slice_name, (start, end) in SLICES.items():
                    item = make_scaled(name, source, scale, cap, slice_name, start, end)
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
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

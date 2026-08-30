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
SOURCE_DIR = REPORT_DIR / "signals" / "core_plus_micro_overlay"
SIGNAL_DIR = REPORT_DIR / "signals" / "micro_overlay_slice_admission"
LOG_DIR = REPORT_DIR / "logs" / "micro_overlay_slice_admission"
OUT_CSV = REPORT_DIR / "micro_overlay_slice_admission_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "micro_overlay_slice_admission_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "微仓补充候选年度切片准入复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CANDIDATES = {
    "bl_empty_o044_s005": SOURCE_DIR / "bl_empty_o044_s005.csv",
    "gb_lowsum_o044_s005": SOURCE_DIR / "gb_lowsum_o044_s005.csv",
}

SLICES = {
    "slice_2024": ("20240101", "20241231"),
    "slice_2025": ("20250101", "20251231"),
    "slice_2026": ("20260101", "20261231"),
    "slice_2025_2026": ("20250101", "20261231"),
    "slice_from_202501": ("20250101", "20991231"),
    "slice_from_202407": ("20240701", "20991231"),
}


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    return df


def write_slice(name: str, source: pd.DataFrame, slice_name: str, start: str, end: str) -> dict | None:
    df = source[(source["buy_date"] >= start) & (source["buy_date"] <= end)].copy()
    if df.empty:
        return None
    out = SIGNAL_DIR / f"{name}_{slice_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    max_positions = int(max(df.groupby("buy_date")["stock_code"].count().max(), 1))
    return {
        "case": name,
        "slice": slice_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "max_positions": max_positions,
        "holding_days": 1,
        "max_holding_days": 3,
        "min_buy_date": str(df["buy_date"].min()),
        "max_buy_date": str(df["buy_date"].max()),
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}_{row['slice']}_mp{row['max_positions']}.log"
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
    lines = [
        "# 微仓补充候选年度切片准入复核",
        "",
        "## 结论",
        "",
        "- 本轮复核 `bl_empty_o044_s005` 与 `gb_lowsum_o044_s005` 的独立年度和近期启动表现。",
        "- 这些结果用于判断表面达标候选是否真正降低路径依赖；不能仅凭全周期指标准入。",
        "",
        "## 掘金切片结果",
        "",
        "| case | slice | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = frame.sort_values(["case", "slice"])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('slice')} | "
            f"{float(row.get('pnl_ratio_annual', 0))*100:.2f}% | {float(row.get('sharp_ratio', 0)):.3f} | "
            f"{float(row.get('max_drawdown', 0))*100:.2f}% | {int(float(row.get('open_count', 0)))} | "
            f"{int(float(row.get('buy_days', 0)))} |"
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
    for name, path in CANDIDATES.items():
        source = load_signal(path)
        for slice_name, (start, end) in SLICES.items():
            item = write_slice(name, source, slice_name, start, end)
            if item is not None:
                manifests.append(item)
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
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "report": str(REPORT_MD), "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

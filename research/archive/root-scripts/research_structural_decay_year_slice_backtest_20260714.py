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
SOURCE = REPORT_DIR / "signals" / "core_structural_decay_refine" / "x125_drop_bigmv_lowturn.csv"
SIGNAL_DIR = REPORT_DIR / "signals" / "structural_decay_year_slices"
LOG_DIR = REPORT_DIR / "logs" / "structural_decay_year_slices"
OUT_CSV = REPORT_DIR / "structural_decay_year_slice_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "structural_decay_year_slice_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "结构衰减修正候选年度切片复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


SLICES = [
    ("slice_2024", "20240101", "20241231"),
    ("slice_2025", "20250101", "20251231"),
    ("slice_2026", "20260101", "20261231"),
    ("slice_2025_2026", "20250101", "20261231"),
    ("slice_from_202407", "20240701", "20261231"),
    ("slice_from_202501", "20250101", "20261231"),
]


def load_source() -> pd.DataFrame:
    return pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})


def write_slice(df: pd.DataFrame, name: str, start: str, end: str) -> dict:
    part = df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()
    part["strategy_variant"] = f"x125_drop_bigmv_lowturn_{name}"
    part["filter_name"] = f"x125_drop_bigmv_lowturn_{name}"
    out = SIGNAL_DIR / f"x125_drop_bigmv_lowturn_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(out, index=False, encoding="utf-8-sig")
    return {
        "case": f"x125_drop_bigmv_lowturn_{name}",
        "slice_start": start,
        "slice_end": end,
        "signal_file": str(out),
        "rows": int(len(part)),
        "buy_days": int(part["buy_date"].nunique()) if len(part) else 0,
        "stock_count": int(part["stock_code"].nunique()) if len(part) else 0,
        "max_positions": 1,
        "holding_days": 1,
        "max_holding_days": 3,
        "min_buy_date": str(part["buy_date"].min()) if len(part) else None,
        "max_buy_date": str(part["buy_date"].max()) if len(part) else None,
    }


def run_or_parse(row: dict) -> dict:
    log = LOG_DIR / f"{row['case']}.log"
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
        "--strategy-dir", str(base_mod.STRATEGY_DIR),
        "--signal-file", row["signal_file"],
        "--log-file", str(log),
        "--score-db", str(base_mod.SCORE_DB),
        "--score-table", base_mod.SCORE_TABLE,
        "--market-db", str(base_mod.MARKET_DB),
        "--max-positions", "1",
        "--holding-days", "1",
        "--max-holding-days", "3",
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    ind = None
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
            ind = payload.get("indicator") or {}
            if isinstance(ind, str):
                ind = base_mod.parse_indicator_text(ind)
        except Exception:
            ind = base_mod.extract_indicator(text)
    else:
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
        "# 结构衰减修正候选年度切片复核",
        "",
        "## 结论",
        "",
        "- 本报告用于区分“去掉 2024 的空档压力测试”和“后期独立启动表现”。",
        "- 回测窗口由切片信号的最小/最大 `buy_date` 自动推断，代表对应时间段独立启动。",
        "",
        "## 掘金切片结果",
        "",
        "| case | 信号区间 | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.sort_values("min_buy_date").iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('min_buy_date')}-{row.get('max_buy_date')} | "
            f"{float(row.get('pnl_ratio_annual', 0))*100:.2f}% | {float(row.get('sharp_ratio', 0)):.3f} | "
            f"{float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | {int(row.get('buy_days', 0))} |"
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
    source = load_source()
    manifests = [write_slice(source, name, start, end) for name, start, end in SLICES]
    results = []
    for row in manifests:
        if row["rows"] == 0:
            results.append({**row, "returncode": None, "indicator_error": "empty_slice"})
            continue
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame)
    print(json.dumps({"csv": str(OUT_CSV), "json": str(OUT_JSON), "report": str(REPORT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

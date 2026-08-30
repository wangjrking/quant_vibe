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
SIGNAL_DIR = REPORT_DIR / "signals" / "latest_core_plus_tiny_refill"
LOG_DIR = REPORT_DIR / "logs" / "latest_core_plus_tiny_refill"
OUT_CSV = REPORT_DIR / "latest_core_plus_tiny_refill_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "latest_core_plus_tiny_refill_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "最新覆盖核心加微仓补位复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CORE_FILES = {
    "ogd_deep8": REPORT_DIR / "signals" / "open_gap_deep_rebalance" / "ogd_deep8_up110.csv",
    "cpo_low60": REPORT_DIR / "signals" / "core_plus_pullback_overlay" / "cpo_low60_o06_cap100.csv",
}

REFILL_FILE = REPORT_DIR / "signals" / "lbc_tiny_fallback" / "ltf_t02_strict.csv"

SLICES = {
    "full": ("00000000", "99999999"),
    "from_202407": ("20240701", "20991231"),
    "from_202501": ("20250101", "20991231"),
    "recent60": ("RECENT60", "RECENT60"),
}

CASES = [
    {"core": "ogd_deep8", "core_scale": 2.0, "cap": 1.0, "refill_target": 0.03, "mode": "empty", "max_refill": 1},
    {"core": "ogd_deep8", "core_scale": 2.5, "cap": 1.0, "refill_target": 0.03, "mode": "empty", "max_refill": 1},
    {"core": "ogd_deep8", "core_scale": 2.5, "cap": 1.0, "refill_target": 0.05, "mode": "lowsum", "low_sum": 0.55, "max_refill": 1},
    {"core": "cpo_low60", "core_scale": 2.0, "cap": 1.0, "refill_target": 0.03, "mode": "empty", "max_refill": 1},
    {"core": "cpo_low60", "core_scale": 2.5, "cap": 1.0, "refill_target": 0.03, "mode": "empty", "max_refill": 1},
    {"core": "cpo_low60", "core_scale": 2.5, "cap": 1.0, "refill_target": 0.05, "mode": "lowsum", "low_sum": 0.55, "max_refill": 1},
]


def _num(value: object, default: float = 0.0) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(out):
        return default
    return float(out)


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    df["target_pct"] = pd.to_numeric(df["target_pct"], errors="coerce").fillna(0.0)
    if "sort_score" not in df.columns:
        df["sort_score"] = pd.to_numeric(df.get("entry_score", df.get("pred_prob")), errors="coerce").fillna(0.0)
    else:
        df["sort_score"] = pd.to_numeric(df["sort_score"], errors="coerce").fillna(0.0)
    return df


def slice_signal(df: pd.DataFrame, slice_name: str, start: str, end: str) -> pd.DataFrame:
    if slice_name == "recent60":
        days = sorted(df["buy_date"].dropna().unique())[-60:]
        return df[df["buy_date"].isin(days)].copy()
    return df[(df["buy_date"] >= start) & (df["buy_date"] <= end)].copy()


def build_case(case: dict, slice_name: str, start: str, end: str) -> dict | None:
    core = slice_signal(load_signal(CORE_FILES[str(case["core"])]), slice_name, start, end)
    refill = slice_signal(load_signal(REFILL_FILE), slice_name, start, end)
    if core.empty:
        return None

    core = core.copy()
    core["layer"] = "core"
    core["target_pct"] = core["target_pct"] * float(case["core_scale"])
    core_daily = core.groupby("buy_date")["target_pct"].sum().rename("core_sum")
    refill = refill.merge(core_daily, on="buy_date", how="left")
    refill["core_sum"] = refill["core_sum"].fillna(0.0)
    if case["mode"] == "empty":
        refill = refill[refill["core_sum"] <= 0].copy()
    elif case["mode"] == "lowsum":
        refill = refill[refill["core_sum"] < float(case.get("low_sum", 0.55))].copy()
    else:
        raise ValueError(str(case["mode"]))

    if not refill.empty:
        core_keys = set(zip(core["buy_date"], core["stock_code"]))
        refill = refill[~refill.apply(lambda row: (row["buy_date"], row["stock_code"]) in core_keys, axis=1)].copy()
        refill = (
            refill.sort_values(["buy_date", "sort_score"], ascending=[True, False])
            .groupby("buy_date", group_keys=False)
            .head(int(case["max_refill"]))
            .copy()
        )
        refill["target_pct"] = float(case["refill_target"])
        refill["layer"] = "tiny_refill"

    df = pd.concat([core, refill], ignore_index=True, sort=False)
    df = df.sort_values(["buy_date", "layer", "sort_score"], ascending=[True, True, False]).copy()
    # If the same stock appears twice on a day, keep the higher target row.
    df = (
        df.sort_values(["buy_date", "stock_code", "target_pct"], ascending=[True, True, False])
        .drop_duplicates(["buy_date", "stock_code"], keep="first")
        .sort_values(["buy_date", "sort_score"], ascending=[True, False])
        .copy()
    )
    daily_sum = df.groupby("buy_date")["target_pct"].transform("sum")
    factor = (float(case["cap"]) / daily_sum).clip(upper=1.0)
    df["target_pct"] = (df["target_pct"] * factor).clip(lower=0.0)
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")

    case_name = (
        f"{case['core']}_x{str(case['core_scale']).replace('.', 'p')}"
        f"_rf{int(float(case['refill_target']) * 100):02d}_{case['mode']}_{slice_name}"
    )
    df["strategy_variant"] = case_name
    df["filter_name"] = case_name
    df["rank"] = df.groupby("buy_date")["sort_score"].rank(method="first", ascending=False).astype(int)
    out = SIGNAL_DIR / f"{case_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily = df.groupby("buy_date")["target_pct"].sum()
    return {
        "case": case_name,
        "core": case["core"],
        "core_scale": float(case["core_scale"]),
        "refill_target": float(case["refill_target"]),
        "mode": case["mode"],
        "slice": slice_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "core_rows": int((df["layer"] == "core").sum()),
        "refill_rows": int((df["layer"] == "tiny_refill").sum()),
        "buy_days": int(df["buy_date"].nunique()),
        "refill_buy_days": int(df.loc[df["layer"] == "tiny_refill", "buy_date"].nunique()),
        "stock_count": int(df["stock_code"].nunique()),
        "min_buy": str(df["buy_date"].min()),
        "max_buy": str(df["buy_date"].max()),
        "max_positions": int(max(df.groupby("buy_date")["stock_code"].count().max(), 1)),
        "mean_daily_target_sum": float(daily.mean()),
        "max_daily_target_sum": float(daily.max()),
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
        "# 最新覆盖核心加微仓补位复核",
        "",
        "## 结论",
        "",
        "- 本轮只验证当前最新覆盖核心信号叠加小仓位补位，不修改生产策略。",
        "- 目标是检查补位能否改善 2025-01 后和 recent60 的年化不足。",
        "- 回测指标以掘金日志解析为准。",
        "",
        "## 结果",
        "",
        "| 候选 | 切片 | 年化 | Sharpe | 最大回撤 | 开仓 | 买入日 | 补位日 | 平均目标仓位 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    d = frame.sort_values(["slice", "pnl_ratio_annual", "sharp_ratio"], ascending=[True, False, False])
    for _, row in d.iterrows():
        lines.append(
            f"| {row.get('case')} | {row.get('slice')} | {_num(row.get('pnl_ratio_annual')) * 100:.2f}% | "
            f"{_num(row.get('sharp_ratio')):.3f} | {_num(row.get('max_drawdown')) * 100:.2f}% | "
            f"{int(_num(row.get('open_count')))} | {int(_num(row.get('buy_days')))} | "
            f"{int(_num(row.get('refill_buy_days')))} | {_num(row.get('mean_daily_target_sum')) * 100:.2f}% |"
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
    for case in CASES:
        for slice_name, (start, end) in SLICES.items():
            item = build_case(case, slice_name, start, end)
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

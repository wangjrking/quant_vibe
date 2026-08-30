from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"
SIGNAL_DIR = REPORT_DIR / "signals" / "core_switch_refill"
LOG_DIR = REPORT_DIR / "logs" / "core_switch_refill"
OUT_CSV = REPORT_DIR / "core_switch_refill_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "core_switch_refill_juejin_results_20260714.json"
REPORT_MD = REPORT_DIR / "核心滚动阻断补位切换复核_20260714.md"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CORES = {
    "x125_drop": REPORT_DIR / "signals" / "core_buy_open_gap_variants" / "x125_drop_missing_gap.csv",
    "x115_drop": REPORT_DIR / "signals" / "core_buy_open_gap_variants" / "x115_drop_missing_gap.csv",
}
REFILLS = {
    "iap044": REPORT_DIR / "signals" / "independent_alpha_pockets" / "iap_04404_p10p1_broad_t2_h2.csv",
    "pgrstab": REPORT_DIR / "signals" / "pool_pullback_gap_rule" / "pgr_stab_w10p1_t2_p990_p1p9_pct10_15_gap3_05_pos34.csv",
}


GATES = [
    {"name": "w10_m2_w50", "window": 10, "mean_min": -0.002, "win_min": 0.50, "warmup": 10},
    {"name": "w10_0_w50", "window": 10, "mean_min": 0.0, "win_min": 0.50, "warmup": 10},
    {"name": "w10_2_w55", "window": 10, "mean_min": 0.002, "win_min": 0.55, "warmup": 10},
]
REFILL_SCALES = [0.34, 0.50, 0.68, 0.90]


def load_signal(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in ["target_pct", "sort_score", "rank", "pred_10d", "pred_prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "sort_score" not in df.columns:
        df["sort_score"] = pd.to_numeric(df.get("pred_10d", df.get("pred_prob", 0.0)), errors="coerce").fillna(0.0)
    return df


def top1_by_day(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["buy_date", "sort_score"]
    asc = [True, False]
    if "rank" in df.columns:
        cols.append("rank")
        asc.append(True)
    return df.sort_values(cols, ascending=asc).groupby("buy_date", as_index=False).head(1).copy()


def add_shadow_returns(top: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        prices = con.execute(
            """
            with cal as (
              select trade_date, lead(trade_date, 2) over(order by trade_date) as d2
              from (select distinct trade_date from STOCK_DAILY_DATA order by trade_date)
            )
            select b.stock_code, b.trade_date as buy_date, b.open as buy_open, e2.open as open_d2
            from STOCK_DAILY_DATA b
            left join cal c on c.trade_date=b.trade_date
            left join STOCK_DAILY_DATA e2 on e2.trade_date=c.d2 and e2.stock_code=b.stock_code
            where b.trade_date between '20220607' and '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    prices["buy_date"] = prices["buy_date"].astype(str)
    out = top.merge(prices, on=["stock_code", "buy_date"], how="left")
    out["shadow_ret"] = out["open_d2"] / out["buy_open"] - 1.0
    return out


def allowed_dates(core_top: pd.DataFrame, gate: dict) -> tuple[set[str], pd.DataFrame]:
    rows = []
    history: list[float] = []
    for _, row in core_top.sort_values("buy_date").iterrows():
        if len(history) < int(gate["warmup"]):
            allowed = True
            roll_mean = None
            roll_win = None
        else:
            vals = pd.Series(history[-int(gate["window"]) :])
            roll_mean = float(vals.mean())
            roll_win = float((vals > 0).mean())
            allowed = roll_mean >= float(gate["mean_min"]) and roll_win >= float(gate["win_min"])
        rows.append(
            {
                "buy_date": row["buy_date"],
                "stock_code": row["stock_code"],
                "shadow_ret": row["shadow_ret"],
                "rolling_mean": roll_mean,
                "rolling_win": roll_win,
                "allowed": bool(allowed),
            }
        )
        if pd.notna(row["shadow_ret"]):
            history.append(float(row["shadow_ret"]))
    gate_frame = pd.DataFrame(rows)
    return set(gate_frame.loc[gate_frame["allowed"], "buy_date"].astype(str)), gate_frame


def build_case(core_name: str, core: pd.DataFrame, refill_name: str, refill: pd.DataFrame, gate: dict, refill_scale: float) -> dict:
    core_top = add_shadow_returns(top1_by_day(core))
    allow, gate_frame = allowed_dates(core_top, gate)
    core_days = set(core["buy_date"].astype(str))
    refill_days = core_days - allow

    core_part = core[core["buy_date"].astype(str).isin(allow)].copy()
    refill_part = top1_by_day(refill[refill["buy_date"].astype(str).isin(refill_days)].copy())
    if len(refill_part):
        refill_part["target_pct"] = refill_scale
    name = f"{core_name}_{gate['name']}_refill_{refill_name}_s{int(refill_scale*100):02d}"
    df = pd.concat([core_part, refill_part], ignore_index=True, sort=False)
    df = top1_by_day(df)
    df["strategy_variant"] = name
    df["filter_name"] = name
    df["switch_refill_case"] = name
    out = SIGNAL_DIR / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    daily_target = df.groupby("buy_date")["target_pct"].sum() if len(df) else pd.Series(dtype=float)
    return {
        "case": name,
        "core": core_name,
        "refill": refill_name,
        "signal_file": str(out),
        "rows": int(len(df)),
        "buy_days": int(df["buy_date"].nunique()) if len(df) else 0,
        "stock_count": int(df["stock_code"].nunique()) if len(df) else 0,
        "max_positions": 1,
        "holding_days": 1,
        "max_holding_days": 3,
        "gate": gate["name"],
        "refill_scale": refill_scale,
        "core_allowed_days": int(len(allow)),
        "core_blocked_days": int((~gate_frame["allowed"]).sum()),
        "refill_days_used": int(refill_part["buy_date"].nunique()) if len(refill_part) else 0,
        "mean_daily_target_sum": float(daily_target.mean()) if len(daily_target) else 0.0,
        "max_daily_target_sum": float(daily_target.max()) if len(daily_target) else 0.0,
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


def write_report(frame: pd.DataFrame, hits: pd.DataFrame) -> None:
    best = frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").head(12)
    lines = [
        "# 核心阻断后补位切换复核",
        "",
        "## 结论",
        "",
    ]
    if len(hits):
        lines.append(f"- 本轮有 {len(hits)} 个候选满足表面目标。")
        lines.append("- 但需要继续做去月份、去股票、去 2024 复核后才能判断准入。")
    else:
        lines.append("- 本轮没有找到新的表面达标候选。")
    lines.append("- 与直接叠加补位不同，本轮仅在核心被滚动影子账户阻断时才启用补位，避免正常阶段稀释核心收益。")
    lines.extend(
        [
            "",
            "## 最优结果前 12",
            "",
            "| case | 年化 | Sharpe | 最大回撤 | 开仓 | 核心阻断日 | 补位日 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {row.get('case')} | {float(row.get('pnl_ratio_annual', 0))*100:.2f}% | "
            f"{float(row.get('sharp_ratio', 0)):.3f} | {float(row.get('max_drawdown', 0))*100:.2f}% | "
            f"{int(row.get('open_count', 0)) if pd.notna(row.get('open_count')) else ''} | "
            f"{int(row.get('core_blocked_days', 0))} | {int(row.get('refill_days_used', 0))} |"
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
    cores = {name: load_signal(path) for name, path in CORES.items()}
    refills = {name: load_signal(path) for name, path in REFILLS.items()}
    manifests = []
    for core_name, core in cores.items():
        for refill_name, refill in refills.items():
            for gate in GATES:
                for scale in REFILL_SCALES:
                    manifests.append(build_case(core_name, core, refill_name, refill, gate, scale))
    results = []
    for row in manifests:
        result = run_or_parse(row)
        results.append(result)
        frame = pd.DataFrame(results)
        frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"case": result.get("case"), "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown"), "open": result.get("open_count")}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(results)
    annual = pd.to_numeric(frame["pnl_ratio_annual"], errors="coerce")
    sharpe = pd.to_numeric(frame["sharp_ratio"], errors="coerce")
    mdd = pd.to_numeric(frame["max_drawdown"], errors="coerce")
    hits = frame[(annual >= 5.0) & (sharpe >= 4.0) & (mdd <= 0.4)].copy()
    summary = {
        "cores": {k: str(v) for k, v in CORES.items()},
        "refills": {k: str(v) for k, v in REFILLS.items()},
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "report": str(REPORT_MD),
        "cases": int(len(results)),
        "target_hits": int(len(hits)),
        "best": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").head(12).to_dict("records"),
        "target_hits_table": hits.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last").to_dict("records"),
    }
    (REPORT_DIR / "core_switch_refill_summary_20260714.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(frame, hits)
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

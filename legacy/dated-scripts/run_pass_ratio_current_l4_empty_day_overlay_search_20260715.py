from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "run_pass_ratio_open_verified_scale_fine_20260715.py"

spec = importlib.util.spec_from_file_location("scale_fine", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_pass_ratio_open_gap_sellfreq_20260715"
SOURCE_REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
CORE_SIGNAL = REPORT_DIR / "signals" / "recalc_hardgate_scale" / "rh_top1_s120_cap065_full.csv"
CANDIDATE_POOL = SOURCE_REPORT_DIR / "active_l4_multilabel_pool_candidates_20260714.parquet"
L2_DB = ROOT / "quant" / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"

OUT_DIR = REPORT_DIR / "signals" / "current_l4_empty_day_overlay"
LOG_DIR = REPORT_DIR / "logs" / "current_l4_empty_day_overlay_20260715"
PROXY_CSV = REPORT_DIR / "current_l4_empty_day_overlay_proxy_20260715.csv"
OUT_CSV = REPORT_DIR / "current_l4_empty_day_overlay_juejin_20260715.csv"
OUT_JSON = REPORT_DIR / "current_l4_empty_day_overlay_juejin_20260715.json"
OUT_MD = REPORT_DIR / "current_l4_empty_day_overlay_review_20260715.md"

TRADE_DAYS = 997


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".", 1)
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def load_core() -> pd.DataFrame:
    core = pd.read_csv(CORE_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    core["target_pct"] = pd.to_numeric(core["target_pct"], errors="coerce").fillna(0.0)
    core["coverage_layer"] = "core"
    return core


def load_pool(core_buy_dates: set[str]) -> pd.DataFrame:
    pool = pd.read_parquet(CANDIDATE_POOL)
    for col in [
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "buy_open_gap_raw_pct",
        "buy_open_gap_pct",
    ]:
        pool[col] = pd.to_numeric(pool[col], errors="coerce")
    pool["signal_date"] = pool["signal_date"].astype(str)
    pool["buy_date"] = pool["buy_date"].astype(str)
    pool = pool[~pool["buy_date"].isin(core_buy_dates)].copy()
    pool = pool[~pool["stock_code"].astype(str).str.endswith(".BJ")].copy()
    pool = pool[pool["ST_TYPE"].fillna("").astype(str).isin(["", "0", "0.0", "None", "nan"])].copy()
    pool = pool[~pool["name"].fillna("").astype(str).str.startswith(("ST", "*ST", "退"))].copy()
    pool = pool[
        (pool["pred_10d"] >= 0.94)
        & (pool["amount"] >= 80_000)
        & (pool["atr_qfq"] <= 30.0)
        & (pool["signal_pct_chg_raw"] >= -12.0)
        & (pool["signal_pct_chg_raw"] <= 2.0)
        & (pool["buy_open_gap_raw_pct"] >= -8.0)
        & (pool["buy_open_gap_raw_pct"] <= 1.5)
    ].copy()
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        pool[f"{col}_rank"] = pool.groupby("signal_date")[col].rank(method="average", pct=True)
    return pool


def load_next_open_returns() -> pd.DataFrame:
    con = duckdb.connect(str(L2_DB), read_only=True)
    try:
        ret = con.execute(
            """
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) over(order by trade_date) AS next_date
                FROM (SELECT DISTINCT trade_date FROM STOCK_DAILY_DATA ORDER BY trade_date)
            )
            SELECT
                b.stock_code,
                b.trade_date AS buy_date,
                b.open AS buy_open_raw,
                n.open AS next_open_raw,
                (n.open / b.open - 1.0) AS next_open_ret
            FROM STOCK_DAILY_DATA b
            JOIN cal c ON c.trade_date = b.trade_date
            LEFT JOIN STOCK_DAILY_DATA n ON n.trade_date = c.next_date AND n.stock_code = b.stock_code
            WHERE b.trade_date BETWEEN '20220607' AND '20260714'
            """
        ).fetchdf()
    finally:
        con.close()
    ret["buy_date"] = ret["buy_date"].astype(str)
    return ret


def make_cases() -> list[dict]:
    weights = {
        "w10_1d": (0.25, 0.00, 0.00, 0.75),
        "w10_5d": (0.00, 0.00, 0.25, 0.75),
    }
    cases: list[dict] = []
    idx = 0
    for w_name, w in weights.items():
        for top_n, target, p10_min, score_min, pct_lo, pct_hi, gap_lo, gap_hi, amt_min, turn_min, atr_max in itertools.product(
            [1],
            [0.05, 0.08, 0.12],
            [0.94, 0.97],
            [0.75, 0.85],
            [-12.0, -8.0, -5.0],
            [-1.5, 0.0],
            [-8.0, -5.0],
            [0.5, 1.5],
            [120_000, 250_000],
            [0.0, 1.5],
            [15.0],
        ):
            if pct_lo >= pct_hi:
                continue
            idx += 1
            cases.append(
                {
                    "case": f"edo_{idx:05d}_{w_name}_t{top_n}_p{int(target*100):02d}",
                    "weights": w,
                    "top_n": top_n,
                    "target": target,
                    "p10_min": p10_min,
                    "score_min": score_min,
                    "pct_lo": pct_lo,
                    "pct_hi": pct_hi,
                    "gap_lo": gap_lo,
                    "gap_hi": gap_hi,
                    "amount_min": amt_min,
                    "turn_min": turn_min,
                    "atr_max": atr_max,
                }
            )
    return cases


def select_overlay(pool: pd.DataFrame, case: dict) -> pd.DataFrame:
    w1, w3, w5, w10 = case["weights"]
    work = pool.copy()
    work["overlay_score"] = (
        w1 * work["pred_1d_rank"]
        + w3 * work["pred_3d_rank"]
        + w5 * work["pred_5d_rank"]
        + w10 * work["pred_10d_rank"]
    )
    mask = (
        (work["pred_10d"] >= case["p10_min"])
        & (work["overlay_score"] >= case["score_min"])
        & (work["signal_pct_chg_raw"] >= case["pct_lo"])
        & (work["signal_pct_chg_raw"] <= case["pct_hi"])
        & (work["buy_open_gap_raw_pct"] >= case["gap_lo"])
        & (work["buy_open_gap_raw_pct"] <= case["gap_hi"])
        & (work["amount"] >= case["amount_min"])
        & (work["turnover_rate"] >= case["turn_min"])
        & (work["atr_qfq"] <= case["atr_max"])
    )
    selected = work.loc[mask].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(
        ["buy_date", "overlay_score", "pred_10d", "amount"],
        ascending=[True, False, False, False],
    )
    selected = selected.groupby("buy_date", group_keys=False).head(int(case["top_n"])).copy()
    selected["rank"] = selected.groupby("buy_date").cumcount() + 1
    selected["target_pct"] = float(case["target"])
    selected["coverage_layer"] = "empty_day_overlay"
    selected["sort_score"] = selected["overlay_score"]
    return selected


def proxy_score(overlay: pd.DataFrame, ret: pd.DataFrame, case: dict) -> dict | None:
    if overlay.empty or overlay["buy_date"].nunique() < 40:
        return None
    joined = overlay.merge(ret[["stock_code", "buy_date", "next_open_ret"]], on=["stock_code", "buy_date"], how="left")
    joined = joined.dropna(subset=["next_open_ret"]).copy()
    if joined.empty or joined["buy_date"].nunique() < 40:
        return None
    joined["weighted_ret"] = joined["target_pct"] * joined["next_open_ret"]
    daily = joined.groupby("buy_date", as_index=False).agg(daily_ret=("weighted_ret", "sum"), names=("stock_code", "count"))
    total = float(daily["daily_ret"].sum())
    std = daily["daily_ret"].std()
    sharpe = float(daily["daily_ret"].mean() / std * (244 ** 0.5)) if std else 0.0
    daily["year"] = daily["buy_date"].str.slice(0, 4)
    by_year = daily.groupby("year")["daily_ret"].sum()
    recent60 = float(daily.tail(60)["daily_ret"].sum()) if len(daily) >= 60 else total
    min_year = float(by_year.min()) if len(by_year) else 0.0
    objective = total + 0.7 * sharpe + 1.5 * min_year + 0.8 * recent60
    return {
        "case": case["case"],
        "overlay_rows": int(len(overlay)),
        "overlay_buy_days": int(overlay["buy_date"].nunique()),
        "overlay_latest_buy_date": str(overlay["buy_date"].max()),
        "proxy_total": total,
        "proxy_sharpe": sharpe,
        "proxy_recent60": recent60,
        "proxy_min_year": min_year,
        "objective": objective,
        **{k: v for k, v in case.items() if k != "weights"},
        "w1": case["weights"][0],
        "w3": case["weights"][1],
        "w5": case["weights"][2],
        "w10": case["weights"][3],
    }


def normalize_overlay(overlay: pd.DataFrame, case_name: str) -> pd.DataFrame:
    df = overlay.copy()
    df["symbol"] = df["stock_code"].map(to_symbol)
    df["pred_prob"] = df["overlay_score"]
    df["entry_score"] = df["overlay_score"]
    df["holding_days"] = 1
    df["max_holding_days"] = 3
    df["score_exit_entry_ratio"] = "0.98000"
    df["min_holding_days_before_score_exit"] = 1
    df["score_continue_entry_ratio"] = "1.02000"
    df["signal_stop_loss_pct"] = 0.05
    df["signal_take_profit_pct"] = 0.08
    df["strategy_variant"] = case_name
    df["source_strategy_variant"] = "active_l4_empty_day_overlay_current_research"
    df["filter_name"] = case_name
    df["entry_weight_name"] = "current_l4_empty_day_overlay"
    df["dynamic_hold_name"] = "h1_mh3_score_exit098"
    df["buy_day_market_available"] = True
    df["buy_day_hard_gate_complete"] = True
    df["buy_day_st_rejected"] = False
    df["buy_day_open_limit_up_rejected"] = False
    df["latest_market_date"] = "20260714"
    df["hybrid_source"] = "current_active_formal_l4_empty_day_overlay"
    df["feature_weight_scale"] = 1.0
    df["daily_target_sum_after_cap"] = df.groupby("buy_date")["target_pct"].transform("sum")
    return df


def build_signal(core: pd.DataFrame, overlay: pd.DataFrame, case_name: str) -> pd.DataFrame:
    overlay_norm = normalize_overlay(overlay, case_name)
    core_norm = core.copy()
    core_norm["strategy_variant"] = case_name
    core_norm["filter_name"] = case_name
    core_norm["coverage_layer"] = "core"
    combined = pd.concat([core_norm, overlay_norm], ignore_index=True, sort=False)
    combined = combined.sort_values(["buy_date", "coverage_layer", "rank"]).copy()
    combined["rank"] = combined.groupby("buy_date").cumcount() + 1
    combined["daily_target_sum_after_cap"] = combined.groupby("buy_date")["target_pct"].transform("sum")
    cols = [
        "signal_date",
        "buy_date",
        "symbol",
        "stock_code",
        "name",
        "rank",
        "pred_prob",
        "entry_score",
        "pred_1d",
        "pred_3d",
        "pred_5d",
        "pred_10d",
        "amount",
        "turnover_rate",
        "total_mv",
        "atr_qfq",
        "signal_pct_chg_raw",
        "target_pct",
        "holding_days",
        "max_holding_days",
        "score_exit_entry_ratio",
        "min_holding_days_before_score_exit",
        "score_continue_entry_ratio",
        "signal_stop_loss_pct",
        "signal_take_profit_pct",
        "strategy_variant",
        "source_strategy_variant",
        "filter_name",
        "entry_weight_name",
        "dynamic_hold_name",
        "buy_day_market_available",
        "buy_day_hard_gate_complete",
        "buy_day_st_rejected",
        "buy_day_open_limit_up_rejected",
        "latest_market_date",
        "buy_open_gap_pct",
        "hybrid_source",
        "buy_open_gap_raw_pct",
        "feature_weight_scale",
        "daily_target_sum_after_cap",
        "coverage_layer",
    ]
    for col in cols:
        if col not in combined.columns:
            combined[col] = None
    return combined[cols].copy()


def slice_frame(frame: pd.DataFrame, start: str | None) -> pd.DataFrame:
    if start == "__recent60__":
        dates = sorted(frame["buy_date"].dropna().astype(str).unique())
        keep = set(dates[-60:])
        return frame[frame["buy_date"].astype(str).isin(keep)].copy()
    if start:
        return frame[frame["buy_date"].astype(str) >= start].copy()
    return frame.copy()


def run_juejin(signal_file: Path, case: str, slice_name: str, max_positions: int) -> dict:
    log_file = LOG_DIR / f"{case}_{slice_name}.log"
    cmd = [
        sys.executable,
        str(base.RUNNER),
        "--strategy-dir",
        str(base.STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(max_positions),
        "--score-db",
        str(base.SCORE_DB),
        "--score-table",
        base.SCORE_TABLE,
        "--market-db",
        str(base.MARKET_DB),
    ]
    proc = subprocess.run(cmd, cwd=str(MAIN), capture_output=True, text=True)
    text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else ""
    metrics = base.parse_metrics(text)
    return {
        "case": case,
        "slice": slice_name,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "returncode": proc.returncode,
        **metrics,
    }


def pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return ""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    core = load_core()
    core_dates = set(core["buy_date"].astype(str))
    pool = load_pool(core_dates)
    returns = load_next_open_returns()

    proxy_rows: list[dict] = []
    overlay_cache: dict[str, pd.DataFrame] = {}
    case_cache: dict[str, dict] = {}
    for case in make_cases():
        overlay = select_overlay(pool, case)
        score = proxy_score(overlay, returns, case)
        if score is None:
            continue
        proxy_rows.append(score)
        overlay_cache[case["case"]] = overlay
        case_cache[case["case"]] = case

    proxy = pd.DataFrame(proxy_rows).sort_values("objective", ascending=False)
    proxy.to_csv(PROXY_CSV, index=False, encoding="utf-8-sig")
    selected_cases = proxy.head(6)["case"].tolist()

    rows: list[dict] = []
    for case_name in selected_cases:
        overlay = overlay_cache[case_name]
        signal = build_signal(core, overlay, case_name)
        max_positions = int(signal.groupby("buy_date")["stock_code"].count().max()) if len(signal) else 1
        for slice_name, start in base.SLICES:
            sliced = slice_frame(signal, start)
            path = OUT_DIR / f"{case_name}_{slice_name}.csv"
            sliced.to_csv(path, index=False, encoding="utf-8-sig")
            result = run_juejin(path, case_name, slice_name, max_positions)
            result.update(
                {
                    "rows": int(len(sliced)),
                    "buy_days": int(sliced["buy_date"].nunique()) if len(sliced) else 0,
                    "trade_day_coverage": int(sliced["buy_date"].nunique()) / TRADE_DAYS if slice_name == "full" else None,
                    "overlay_rows": int((sliced["coverage_layer"] == "empty_day_overlay").sum()) if len(sliced) else 0,
                    "overlay_buy_days": int(sliced.loc[sliced["coverage_layer"] == "empty_day_overlay", "buy_date"].nunique()) if len(sliced) else 0,
                    "latest_buy_date": str(sliced["buy_date"].max()) if len(sliced) else None,
                    "max_positions": max_positions,
                }
            )
            rows.append(result)
            print(json.dumps({"case": case_name, "slice": slice_name, "buy_days": result["buy_days"], "annual": result.get("pnl_ratio_annual"), "sharpe": result.get("sharp_ratio"), "mdd": result.get("max_drawdown")}, ensure_ascii=False), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    full = frame[frame["slice"] == "full"].copy()
    full = full.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last")
    lines = [
        "# 当前 L4 空窗覆盖层搜索 20260715",
        "",
        "## 口径",
        "",
        "- 核心信号使用 `rh_top1_s120_cap065_full.csv`，覆盖到 buy_date=20260714。",
        "- 覆盖层只允许在核心信号空窗日生成，不做买入日失败后的候补。",
        "- 覆盖层候选来自 active formal L4 候选池，硬过滤使用信号日前可见字段和买入日未复权开盘硬门槛。",
        "- 参数先用次日开盘收益 proxy 排序，再只对前 6 个进入掘金切片复跑。",
        "",
        "## full 结果",
        "",
        "| 版本 | 覆盖率 | 买入日 | 覆盖层买入日 | 年化 | Sharpe | 最大回撤 | 最新买入日 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in full.head(12).iterrows():
        lines.append(
            f"| `{row['case']}` | {pct(row.get('trade_day_coverage'))} | {int(row.get('buy_days', 0))} | "
            f"{int(row.get('overlay_buy_days', 0))} | {pct(row.get('pnl_ratio_annual'))} | "
            f"{float(row.get('sharp_ratio', 0)):.4f} | {pct(row.get('max_drawdown'))} | {row.get('latest_buy_date')} |"
        )
    lines.extend(
        [
            "",
            "## 证据路径",
            "",
            f"- proxy 结果：`{PROXY_CSV}`",
            f"- 掘金结果 CSV：`{OUT_CSV}`",
            f"- 掘金结果 JSON：`{OUT_JSON}`",
            f"- 信号目录：`{OUT_DIR}`",
            f"- 掘金日志目录：`{LOG_DIR}`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)


if __name__ == "__main__":
    main()

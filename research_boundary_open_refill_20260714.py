from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
BASE_SCRIPT = MAIN / "research_top1_three_tier_open_quality_20260714.py"
REPORT_DIR = ROOT / "quant" / "data_file" / "reports" / "strategy_agent_four_year_l4_frequency_optimization_20260714"
BASE_SIGNAL = REPORT_DIR / "signals" / "boundary_lowbase_highboost" / "blhb_o0925_h1040.csv"
POOL_FILE = REPORT_DIR / "active_l4_wide_buy_quality_base_pool_20260714.parquet"
SIGNAL_DIR = REPORT_DIR / "signals" / "boundary_open_refill"
LOG_DIR = REPORT_DIR / "logs" / "boundary_open_refill"
OUT_CSV = REPORT_DIR / "boundary_open_refill_juejin_results_20260714.csv"
OUT_JSON = REPORT_DIR / "boundary_open_refill_juejin_results_20260714.json"


spec = importlib.util.spec_from_file_location("base_three_tier", BASE_SCRIPT)
base_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base_mod)


CASES = [
    {
        "case": "bor_gap05_refill5_liq00",
        "fail_gap_gt": 0.5,
        "fail_shallow_gt": None,
        "refill_top": 5,
        "pct_max": -1.75,
        "gap_max": 0.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.00,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap00_refill5_liq00",
        "fail_gap_gt": 0.0,
        "fail_shallow_gt": None,
        "refill_top": 5,
        "pct_max": -1.75,
        "gap_max": 0.0,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.00,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap05_shallow25_refill5_liq00",
        "fail_gap_gt": 0.5,
        "fail_shallow_gt": -2.5,
        "refill_top": 5,
        "pct_max": -2.5,
        "gap_max": 0.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.00,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap05_refill8_liq02",
        "fail_gap_gt": 0.5,
        "fail_shallow_gt": None,
        "refill_top": 8,
        "pct_max": -1.75,
        "gap_max": 0.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.02,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap00_refill8_liq02",
        "fail_gap_gt": 0.0,
        "fail_shallow_gt": None,
        "refill_top": 8,
        "pct_max": -1.75,
        "gap_max": 0.0,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.02,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap05_shallow25_refill8_liq02",
        "fail_gap_gt": 0.5,
        "fail_shallow_gt": -2.5,
        "refill_top": 8,
        "pct_max": -2.5,
        "gap_max": 0.5,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.02,
        "gap_penalty": 0.015,
    },
    {
        "case": "bor_gap05_refill12_liq04",
        "fail_gap_gt": 0.5,
        "fail_shallow_gt": None,
        "refill_top": 12,
        "pct_max": -1.75,
        "gap_max": 0.5,
        "amount_min": 200000,
        "mv_min": 300000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.04,
        "gap_penalty": 0.020,
    },
    {
        "case": "bor_gap10_refill8_liq02",
        "fail_gap_gt": 1.0,
        "fail_shallow_gt": None,
        "refill_top": 8,
        "pct_max": -1.75,
        "gap_max": 1.0,
        "amount_min": 120000,
        "mv_min": 200000,
        "pred10_rank_min": 0.70,
        "liq_bonus": 0.02,
        "gap_penalty": 0.010,
    },
]


def to_symbol(stock_code: str) -> str:
    code, suffix = str(stock_code).split(".")
    return {"SH": "SHSE", "SZ": "SZSE"}.get(suffix, suffix) + "." + code


def read_base_signal() -> pd.DataFrame:
    df = pd.read_csv(BASE_SIGNAL, encoding="utf-8-sig", dtype={"signal_date": str, "buy_date": str, "stock_code": str})
    for col in [
        "rank",
        "target_pct",
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
        "exec_open_gap_pct",
        "buy_open_gap_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "exec_open_gap_pct" not in df.columns:
        df["exec_open_gap_pct"] = df["buy_open_gap_pct"]
    return df


def read_pool() -> pd.DataFrame:
    df = pd.read_parquet(POOL_FILE)
    df["signal_date"] = df["signal_date"].astype(str)
    df["buy_date"] = df["buy_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d", "amount", "turnover_rate", "total_mv", "signal_pct_chg_raw", "buy_open_gap_raw_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        df[f"{col}_rank"] = df.groupby("signal_date")[col].rank(method="average", pct=True)
    df["blend_score"] = (
        0.25 * df["pred_1d_rank"]
        + 0.25 * df["pred_3d_rank"]
        + 0.50 * df["pred_10d_rank"]
    )
    df["amount_rank"] = df.groupby("signal_date")["amount"].rank(method="average", pct=True)
    return df


def primary_pass(row: pd.Series, case: dict) -> bool:
    gap = float(row.get("exec_open_gap_pct") or 0.0)
    pct = float(row.get("signal_pct_chg_raw") or 0.0)
    if case["fail_gap_gt"] is not None and gap > float(case["fail_gap_gt"]):
        return False
    if case["fail_shallow_gt"] is not None and pct > float(case["fail_shallow_gt"]):
        return False
    return True


def candidate_pool(pool: pd.DataFrame, buy_date: str, case: dict, used: set[str]) -> pd.DataFrame:
    cand = pool[
        (pool["buy_date"] == buy_date)
        & (~pool["stock_code"].isin(used))
        & (pool["signal_pct_chg_raw"] <= case["pct_max"])
        & (pool["buy_open_gap_raw_pct"] <= case["gap_max"])
        & (pool["buy_open_gap_raw_pct"] >= -8.0)
        & (pool["amount"] >= case["amount_min"])
        & (pool["total_mv"] >= case["mv_min"])
        & (pool["pred_10d_rank"] >= case["pred10_rank_min"])
        & (~pool["buy_open_limit_up_rejected"].fillna(False))
    ].copy()
    if cand.empty:
        return cand
    cand["refill_score"] = (
        cand["blend_score"]
        + float(case["liq_bonus"]) * cand["amount_rank"].fillna(0.0)
        - float(case["gap_penalty"]) * cand["buy_open_gap_raw_pct"].clip(lower=0).fillna(0.0)
    )
    return cand.sort_values(["refill_score", "pred_10d_rank", "amount"], ascending=[False, False, False]).head(int(case["refill_top"]))


def row_from_candidate(candidate: pd.Series, template: pd.Series, rank: int, case_name: str) -> dict:
    row = {col: template.get(col) for col in template.index}
    row.update(
        {
            "signal_date": str(candidate["signal_date"]),
            "buy_date": str(candidate["buy_date"]),
            "symbol": to_symbol(str(candidate["stock_code"])),
            "stock_code": str(candidate["stock_code"]),
            "name": candidate.get("name"),
            "rank": rank,
            "pred_prob": float(candidate["blend_score"]),
            "entry_score": float(candidate["blend_score"]),
            "pred_1d": float(candidate["pred_1d_rank"]),
            "pred_3d": float(candidate["pred_3d_rank"]),
            "pred_5d": float(candidate["pred_5d_rank"]),
            "pred_10d": float(candidate["pred_10d_rank"]),
            "amount": float(candidate["amount"]),
            "turnover_rate": float(candidate["turnover_rate"]),
            "total_mv": float(candidate["total_mv"]),
            "atr_qfq": np.nan,
            "signal_pct_chg_raw": float(candidate["signal_pct_chg_raw"]),
            "strategy_variant": case_name,
            "source_strategy_variant": "active_l4_formal_duckdb_open_refill",
            "filter_name": case_name,
            "entry_weight_name": "w25_25_00_50_reconstructed_refill",
            "dynamic_hold_name": "h1m3_refill",
            "buy_day_market_available": True,
            "buy_day_hard_gate_complete": True,
            "buy_day_st_rejected": False,
            "buy_day_open_limit_up_rejected": False,
            "latest_market_date": "20260713",
            "buy_open_gap_pct": float(candidate["buy_open_gap_raw_pct"]),
            "hybrid_source": "wide_pool_refill",
            "buy_open_gap_raw_pct": float(candidate["buy_open_gap_raw_pct"]),
            "exec_open_gap_pct": float(candidate["buy_open_gap_raw_pct"]),
            "quality_bucket": 1,
            "sort_score": float(candidate["refill_score"]),
        }
    )
    return row


def build_variant(base: pd.DataFrame, pool: pd.DataFrame, case: dict) -> dict:
    rows: list[dict] = []
    replaced = 0
    failed_without_refill = 0
    base = base.sort_values(["buy_date", "rank", "entry_score"], ascending=[True, True, False])
    for buy_date, group in base.groupby("buy_date", sort=True):
        used: set[str] = set()
        day_rows: list[dict] = []
        fail_templates: list[pd.Series] = []
        target_n = len(group)
        for _, item in group.iterrows():
            if primary_pass(item, case):
                day_rows.append(item.to_dict())
                used.add(str(item["stock_code"]))
            else:
                fail_templates.append(item)
        for template in fail_templates:
            cand = candidate_pool(pool, str(buy_date), case, used)
            if cand.empty:
                failed_without_refill += 1
                continue
            chosen = cand.iloc[0]
            used.add(str(chosen["stock_code"]))
            day_rows.append(row_from_candidate(chosen, template, len(day_rows) + 1, case["case"]))
            replaced += 1
        day_rows = sorted(day_rows, key=lambda x: (int(float(x.get("rank", 999))), -float(x.get("entry_score") or 0.0)))[:target_n]
        for idx, row in enumerate(day_rows, start=1):
            row["rank"] = idx
        rows.extend(day_rows)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(case["case"] + " produced no rows")
    out["strategy_variant"] = case["case"]
    out["filter_name"] = case["case"]
    out["daily_target_sum_after_cap"] = out.groupby("buy_date")["target_pct"].transform("sum")
    out_path = SIGNAL_DIR / f"{case['case']}.csv"
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "case": case["case"],
        "signal_file": str(out_path),
        "rows": int(len(out)),
        "buy_days": int(out["buy_date"].nunique()),
        "stock_count": int(out["stock_code"].nunique()),
        "replaced_rows": int(replaced),
        "failed_without_refill": int(failed_without_refill),
        "mean_daily_target_sum": float(out.groupby("buy_date")["target_pct"].sum().mean()),
        **case,
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
    return base_mod.run_juejin(row)


def main() -> None:
    base_mod.SIGNAL_DIR = SIGNAL_DIR
    base_mod.LOG_DIR = LOG_DIR
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base = read_base_signal()
    pool = read_pool()
    manifest = [build_variant(base, pool, case) for case in CASES]
    results = []
    for row in manifest:
        result = run_or_parse(row)
        results.append(result)
        pd.DataFrame(results).sort_values(
            ["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False], na_position="last"
        ).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "case": result.get("case"),
                    "annual": result.get("pnl_ratio_annual"),
                    "sharpe": result.get("sharp_ratio"),
                    "mdd": result.get("max_drawdown"),
                    "replaced_rows": result.get("replaced_rows"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    frame = pd.DataFrame(results)
    hits = frame[
        (frame["pnl_ratio_annual"] >= 5.0)
        & (frame["sharp_ratio"] >= 4.0)
        & (frame["max_drawdown"] <= 0.4)
    ]
    summary = {
        "csv": str(OUT_CSV),
        "json": str(OUT_JSON),
        "cases": len(results),
        "target_hits": int(len(hits)),
        "best_by_sharpe": frame.sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(5).to_dict("records"),
        "best_annual_ge_5": frame[frame["pnl_ratio_annual"] >= 5.0].sort_values(["sharp_ratio", "pnl_ratio_annual"], ascending=[False, False]).head(5).to_dict("records"),
    }
    (REPORT_DIR / "boundary_open_refill_summary_20260714.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

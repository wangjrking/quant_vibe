from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(r"D:\work\quant\quant_mcp")
MAIN = ROOT / "quant" / "main"
sys.path.insert(0, str(MAIN))

from prediction_manifest import load_prediction_source_manifest  # noqa: E402
from stock_daily_data_route import resolve_stock_daily_duckdb_path  # noqa: E402


REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_active_l4_prod_repro_grid_20260703"
)
SIGNAL_DIR = REPORT_DIR / "signals"

MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}


@dataclass(frozen=True)
class Case:
    name: str
    weights: tuple[float, float, float, float]
    pct_chg_max: float
    raw_gap_low: float
    raw_gap_high: float
    r10_min: float
    r1_min: float
    amount_min: float
    mv_min: float
    atr_pct_max: float
    topn: int = 3
    target_pct: float = 0.435
    use_liq_rerank: bool = True


def _symbol(stock_code: str) -> str:
    if stock_code.endswith(".SH"):
        return "SHSE." + stock_code[:6]
    if stock_code.endswith(".SZ"):
        return "SZSE." + stock_code[:6]
    return stock_code


def _limit_up_ratio(stock_code: str) -> float:
    if stock_code.startswith(("300", "301", "688")):
        return 0.195
    return 0.095


def _annualized_return(daily_returns: list[float]) -> float:
    if not daily_returns:
        return float("nan")
    equity = 1.0
    for ret in daily_returns:
        equity *= 1.0 + ret
    years = len(daily_returns) / 252.0
    if years <= 0 or equity <= 0:
        return float("nan")
    return equity ** (1.0 / years) - 1.0


def _sharpe(daily_returns: list[float]) -> float:
    if len(daily_returns) < 2:
        return float("nan")
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return float("nan")
    return mean / std * math.sqrt(252.0)


def _max_drawdown(daily_returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for ret in daily_returns:
        equity *= 1.0 + ret
        peak = max(peak, equity)
        if peak > 0:
            mdd = max(mdd, 1.0 - equity / peak)
    return mdd


def _evaluate_local(rows: list[dict]) -> dict:
    by_day: dict[str, list[dict]] = {}
    for row in rows:
        by_day.setdefault(str(row["signal_date"]), []).append(row)
    daily_returns: list[float] = []
    wins = 0
    losses = 0
    for day_rows in by_day.values():
        weights = [float(r["target_pct"]) for r in day_rows]
        weight_sum = sum(weights)
        scale = min(1.0, 1.0 / weight_sum) if weight_sum > 0 else 0.0
        day_ret = 0.0
        for row, weight in zip(day_rows, weights):
            ret = float(row["open_to_next_open_ret"])
            day_ret += weight * scale * ret
            if ret > 0:
                wins += 1
            elif ret < 0:
                losses += 1
        daily_returns.append(day_ret)
    return {
        "local_annual": _annualized_return(daily_returns),
        "local_sharpe": _sharpe(daily_returns),
        "local_max_drawdown": _max_drawdown(daily_returns),
        "local_signal_days": len(by_day),
        "local_open_count": len(rows),
        "local_win_ratio": wins / (wins + losses) if wins + losses else float("nan"),
    }


def _cases() -> list[Case]:
    full = os.environ.get("L5_FULL_GRID") == "1"
    if full:
        weights = [
            ("w25_25_00_50", (0.25, 0.25, 0.0, 0.50)),
            ("w10_15_25_50", (0.10, 0.15, 0.25, 0.50)),
            ("w00_00_30_70", (0.00, 0.00, 0.30, 0.70)),
            ("w00_00_00_100", (0.00, 0.00, 0.00, 1.00)),
            ("w35_15_00_50", (0.35, 0.15, 0.0, 0.50)),
            ("w15_35_00_50", (0.15, 0.35, 0.0, 0.50)),
        ]
        pct_maxes = [-1.0, -1.75, -2.5, -3.5, -5.0]
        raw_gaps = [(-0.08, 0.015), (-0.08, 0.0), (-0.05, 0.015), (-0.05, 0.0), (-0.03, 0.015)]
        r10s = [0.70, 0.80, 0.90]
        r1s = [0.0, 0.60, 0.80]
    else:
        weights = [
            ("w25_25_00_50", (0.25, 0.25, 0.0, 0.50)),
            ("w35_15_00_50", (0.35, 0.15, 0.0, 0.50)),
            ("w15_35_00_50", (0.15, 0.35, 0.0, 0.50)),
        ]
        pct_maxes = [-1.75, -2.5, -3.5]
        raw_gaps = [(-0.08, 0.015), (-0.08, 0.0), (-0.05, 0.015)]
        r10s = [0.70, 0.80]
        r1s = [0.0, 0.60]
    cases: list[Case] = []
    for wname, w in weights:
        for pct in pct_maxes:
            for gap_low, gap_high in raw_gaps:
                for r10 in r10s:
                    for r1 in r1s:
                        name = (
                            f"{wname}_pct{str(pct).replace('-', 'm').replace('.', 'p')}"
                            f"_gap{str(gap_low).replace('-', 'm').replace('.', 'p')}"
                            f"to{str(gap_high).replace('-', 'm').replace('.', 'p')}"
                            f"_r10{int(r10*100)}_r1{int(r1*100)}"
                        )
                        cases.append(
                            Case(
                                name=name,
                                weights=w,
                                pct_chg_max=pct,
                                raw_gap_low=gap_low,
                                raw_gap_high=gap_high,
                                r10_min=r10,
                                r1_min=r1,
                                amount_min=90_000.0,
                                mv_min=200_000.0,
                                atr_pct_max=0.18,
                            )
                        )
    return cases


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        label: load_prediction_source_manifest(path, require_approved=True, allow_legacy=False)
        for label, path in MANIFESTS.items()
    }
    market_db = resolve_stock_daily_duckdb_path(require_exists=True)
    con = duckdb.connect()
    try:
        for label, source in sources.items():
            con.execute(f"ATTACH '{Path(source['db_path']).as_posix()}' AS l4_{label} (READ_ONLY)")
        con.execute(f"ATTACH '{Path(market_db).as_posix()}' AS marketdb (READ_ONLY)")
        tables = {label: sources[label]["table"] for label in sources}
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base AS
            WITH cal AS (
                SELECT
                    trade_date,
                    lead(trade_date) OVER (ORDER BY trade_date) AS buy_date,
                    lead(trade_date, 2) OVER (ORDER BY trade_date) AS sell_date
                FROM (SELECT DISTINCT trade_date FROM marketdb.STOCK_DAILY_DATA)
            ),
            preds AS (
                SELECT
                    p10.trade_date,
                    p10.stock_code,
                    p1.pred_prob AS pred_1d,
                    p3.pred_prob AS pred_3d,
                    p5.pred_prob AS pred_5d,
                    p10.pred_prob AS pred_10d
                FROM l4_10d."{tables['10d']}" p10
                JOIN l4_5d."{tables['5d']}" p5 USING (trade_date, stock_code)
                JOIN l4_3d."{tables['3d']}" p3 USING (trade_date, stock_code)
                JOIN l4_1d."{tables['1d']}" p1 USING (trade_date, stock_code)
                WHERE p10.trade_date BETWEEN '20220606' AND '20260701'
                  AND p10.stock_code NOT LIKE '%.BJ'
            ),
            ranked AS (
                SELECT
                    *,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_1d) AS r1,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_3d) AS r3,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_5d) AS r5,
                    percent_rank() OVER (PARTITION BY trade_date ORDER BY pred_10d) AS r10
                FROM preds
            ),
            md AS (
                SELECT
                    trade_date,
                    stock_code,
                    name,
                    open,
                    close,
                    pre_close,
                    open_qfq,
                    close_qfq,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    ST_TYPE,
                    ST_TYPE_name
                FROM marketdb.STOCK_DAILY_DATA
            )
            SELECT
                r.*,
                cal.buy_date,
                cal.sell_date,
                sig.name,
                sig.amount,
                sig.turnover_rate,
                sig.total_mv,
                sig.atr_qfq,
                sig.pct_chg,
                sig.close AS signal_close_raw,
                sig.close_qfq AS signal_close_qfq,
                buy.open AS buy_open_raw,
                buy.pre_close AS buy_pre_close_raw,
                buy.open_qfq AS buy_open_qfq,
                sell.open AS sell_open_raw,
                buy.open / NULLIF(sig.close, 0) - 1 AS buy_open_gap_raw,
                buy.open_qfq / NULLIF(sig.close_qfq, 0) - 1 AS buy_open_gap_qfq,
                sig.atr_qfq / NULLIF(sig.close_qfq, 0) AS atr_pct_qfq,
                sell.open / NULLIF(buy.open, 0) - 1 AS open_to_next_open_ret,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.amount) AS amount_rank,
                percent_rank() OVER (PARTITION BY r.trade_date ORDER BY sig.total_mv) AS mv_rank,
                percent_rank() OVER (
                    PARTITION BY r.trade_date
                    ORDER BY sig.atr_qfq / NULLIF(sig.close_qfq, 0)
                ) AS atr_rank
            FROM ranked r
            JOIN cal ON cal.trade_date = r.trade_date
            JOIN md sig ON sig.trade_date = r.trade_date AND sig.stock_code = r.stock_code
            JOIN md buy ON buy.trade_date = cal.buy_date AND buy.stock_code = r.stock_code
            JOIN md sell ON sell.trade_date = cal.sell_date AND sell.stock_code = r.stock_code
            WHERE cal.buy_date IS NOT NULL
              AND cal.sell_date IS NOT NULL
              AND coalesce(sig.ST_TYPE, '') IN ('', '0')
              AND coalesce(sig.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND coalesce(sig.name, '') NOT LIKE 'ST%'
              AND coalesce(sig.name, '') NOT LIKE '*ST%'
              AND coalesce(buy.ST_TYPE, '') IN ('', '0')
              AND coalesce(buy.ST_TYPE_name, '') NOT LIKE '%ST%'
              AND sig.close IS NOT NULL
              AND sig.close_qfq IS NOT NULL
              AND buy.open IS NOT NULL
              AND buy.pre_close IS NOT NULL
              AND buy.open_qfq IS NOT NULL
              AND sell.open IS NOT NULL
            """
        )
        base_count = con.execute("SELECT count(*), min(trade_date), max(trade_date) FROM base").fetchone()
        manifest: list[dict] = []
        for case in _cases():
            w1, w3, w5, w10 = case.weights
            score_expr = f"({w1} * r1 + {w3} * r3 + {w5} * r5 + {w10} * r10)"
            rerank_expr = (
                f"{score_expr} + 0.035 * amount_rank + 0.020 * mv_rank - 0.040 * atr_rank"
                if case.use_liq_rerank
                else score_expr
            )
            raw_rows = con.execute(
                f"""
                WITH filtered AS (
                    SELECT
                        *,
                        {score_expr}::DOUBLE AS entry_score,
                        {rerank_expr}::DOUBLE AS rerank_score
                    FROM base
                    WHERE pct_chg <= {case.pct_chg_max}
                      AND buy_open_gap_raw >= {case.raw_gap_low}
                      AND buy_open_gap_raw <= {case.raw_gap_high}
                      AND r10 >= {case.r10_min}
                      AND r1 >= {case.r1_min}
                      AND amount >= {case.amount_min}
                      AND total_mv >= {case.mv_min}
                      AND atr_pct_qfq <= {case.atr_pct_max}
                      AND buy_open_raw / NULLIF(buy_pre_close_raw, 0) - 1 < CASE
                          WHEN stock_code LIKE '300%' OR stock_code LIKE '301%' OR stock_code LIKE '688%' THEN 0.195
                          ELSE 0.095
                      END
                ),
                ranked AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY trade_date
                            ORDER BY rerank_score DESC, stock_code
                        ) AS pick_rank
                    FROM filtered
                )
                SELECT
                    trade_date AS signal_date,
                    buy_date,
                    stock_code,
                    name,
                    pick_rank AS rank,
                    entry_score AS pred_prob,
                    entry_score,
                    pred_1d,
                    pred_3d,
                    pred_5d,
                    pred_10d,
                    r1 AS rank_1d,
                    r3 AS rank_3d,
                    r5 AS rank_5d,
                    r10 AS rank_10d,
                    amount,
                    turnover_rate,
                    total_mv,
                    atr_qfq,
                    pct_chg,
                    buy_open_gap_raw,
                    buy_open_gap_qfq,
                    open_to_next_open_ret
                FROM ranked
                WHERE pick_rank <= {case.topn}
                ORDER BY signal_date, rank
                """
            ).fetchdf()
            if raw_rows.empty:
                continue
            raw_rows.insert(2, "symbol", raw_rows["stock_code"].map(_symbol))
            raw_rows["target_pct"] = f"{case.target_pct:.5f}"
            raw_rows["holding_days"] = 1
            raw_rows["max_holding_days"] = 1
            raw_rows["score_exit_entry_ratio"] = "9.99000"
            raw_rows["min_holding_days_before_score_exit"] = 1
            raw_rows["score_continue_entry_ratio"] = "9.99000"
            raw_rows["signal_stop_loss_pct"] = "0.05000"
            raw_rows["signal_take_profit_pct"] = "0.08000"
            raw_rows["strategy_variant"] = case.name
            raw_rows["filter_name"] = "active_l4_prod_raw_gap_clean"
            raw_rows["entry_weight_name"] = case.name.split("_pct", 1)[0]
            raw_rows["dynamic_hold_name"] = "h1m1_clean_raw_exec"
            raw_rows["buy_day_market_available"] = True
            raw_rows["buy_day_hard_gate_complete"] = True
            raw_rows["buy_day_st_rejected"] = False
            raw_rows["buy_day_open_limit_up_rejected"] = False
            raw_rows["latest_market_date"] = "20260702"
            raw_rows["buy_open_gap_raw_pct"] = raw_rows["buy_open_gap_raw"] * 100.0
            raw_rows["buy_open_gap_pct"] = raw_rows["buy_open_gap_qfq"] * 100.0
            rows = raw_rows.to_dict("records")
            metrics = _evaluate_local(rows)
            signal_file = SIGNAL_DIR / f"{case.name}.csv"
            raw_rows.drop(columns=["buy_open_gap_raw", "buy_open_gap_qfq", "open_to_next_open_ret"]).to_csv(
                signal_file,
                index=False,
                encoding="utf-8",
            )
            manifest.append(
                {
                    "name": case.name,
                    "signal_file": str(signal_file),
                    "rows": int(len(raw_rows)),
                    "signal_days": int(raw_rows["signal_date"].nunique()),
                    "avg_names": float(len(raw_rows) / raw_rows["signal_date"].nunique()),
                    "topn": case.topn,
                    "target_pct": case.target_pct,
                    "weights": case.weights,
                    "pct_chg_max": case.pct_chg_max,
                    "raw_gap_low": case.raw_gap_low,
                    "raw_gap_high": case.raw_gap_high,
                    "r10_min": case.r10_min,
                    "r1_min": case.r1_min,
                    "amount_min": case.amount_min,
                    "mv_min": case.mv_min,
                    "atr_pct_max": case.atr_pct_max,
                    **metrics,
                }
            )
    finally:
        con.close()
    manifest_df = pd.DataFrame(manifest)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            ["local_annual", "local_sharpe", "local_max_drawdown"],
            ascending=[False, False, True],
        )
    manifest_df.to_csv(REPORT_DIR / "local_grid_summary.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "input_manifests": {k: str(v) for k, v in MANIFESTS.items()},
                "market_db": str(market_db),
                "base_count_min_max": base_count,
                "candidate_count": int(len(manifest_df)),
                "note": "research-only; executable filters use raw open/close/pre_close, qfq is explicit only for derived indicators",
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(manifest_df.head(20).to_string(index=False))
    print(REPORT_DIR / "local_grid_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

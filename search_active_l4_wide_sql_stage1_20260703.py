from __future__ import annotations

from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
CACHE_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
OUT_CSV = REPORT_DIR / "active_l4_wide_sql_stage1_summary.csv"
VARIANT_DIR = REPORT_DIR / "active_l4_wide_sql_stage1_variants"

SLIPPAGE = 0.0015
BASELINE_ANNUAL = 0.4853


def metric_from_daily(cal: pd.Series, daily: pd.DataFrame) -> tuple[float, float, float, float, int]:
    s = daily.set_index("buy_date")["daily_return"] if len(daily) else pd.Series(dtype=float)
    s = s.reindex(cal, fill_value=0.0).astype(float)
    arr = s.to_numpy()
    n = len(arr)
    equity = np.cumprod(1 + arr)
    total = equity[-1] - 1 if n else 0.0
    annual = equity[-1] ** (252 / n) - 1 if n and equity[-1] > 0 else -1.0
    std = arr.std(ddof=1) if n > 1 else 0.0
    sharpe = arr.mean() / std * np.sqrt(252) if std > 0 else 0.0
    peak = np.maximum.accumulate(equity) if n else np.array([1.0])
    mdd = -float((equity / peak - 1).min()) if n else 0.0
    return annual, sharpe, mdd, total, int((s != 0).sum())


def sql_literal(v: float | None) -> str:
    return "null" if v is None else str(v)


def main() -> None:
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(CACHE_DB), read_only=True)
    cal = con.execute(
        "select distinct buy_date from active_l4_wide where buy_date between '20220607' and '20260702' order by buy_date"
    ).fetchdf()["buy_date"].astype(str)

    score_mins = [0.80, 0.90, 0.95]
    pct_rules = [
        ("all", None, None),
        ("pull175", None, -1.75),
        ("pull3", None, -3.0),
        ("band12_175", -12.0, -1.75),
    ]
    buy_gap_rules = [
        ("allgap", None, None),
        ("gap_le15", None, 1.5),
        ("gap_m3_15", -3.0, 1.5),
    ]
    amount_mins = [50000, 90000]
    topns = [1, 3, 5]
    holds = [1, 2, 3]
    sort_modes = {
        "score": "pred_prob desc, buy_amount desc",
        "lowgap": "buy_open_gap_raw_pct asc, pred_prob desc",
    }

    rows = []
    candidate_rows = []
    case_id = 0
    for score_min, pct_rule, buy_gap_rule, amount_min, topn, hold, sort_item in product(
        score_mins, pct_rules, buy_gap_rules, amount_mins, topns, holds, sort_modes.items()
    ):
        pct_name, pct_min, pct_max = pct_rule
        gap_name, gap_min, gap_max = buy_gap_rule
        sort_name, order_expr = sort_item
        ret_col = f"ret_h{hold}"
        target_each = min(0.99 / topn, 0.60)
        filters = [
            f"score_pct_rank >= {score_min}",
            f"signal_amount >= {amount_min}",
            f"buy_amount >= {amount_min}",
            f"{ret_col} is not null",
        ]
        if pct_min is not None:
            filters.append(f"signal_pct_chg >= {pct_min}")
        if pct_max is not None:
            filters.append(f"signal_pct_chg <= {pct_max}")
        if gap_min is not None:
            filters.append(f"buy_open_gap_raw_pct >= {gap_min}")
        if gap_max is not None:
            filters.append(f"buy_open_gap_raw_pct <= {gap_max}")
        where = " and ".join(filters)
        query = f"""
            with picked as (
                select
                    *,
                    row_number() over(partition by buy_date order by {order_expr}) as pick_rank
                from active_l4_wide
                where {where}
            )
            select
                buy_date,
                sum((((1 + {ret_col}) * (1 - {SLIPPAGE}) / (1 + {SLIPPAGE}) - 1) * {target_each})) as daily_return,
                count(*) as n_names
            from picked
            where pick_rank <= {topn}
            group by buy_date
            order by buy_date
        """
        daily = con.execute(query).fetchdf()
        if len(daily) < 20:
            continue
        annual, sharpe, mdd, total, active_days = metric_from_daily(cal, daily)
        row = {
            "case_id": case_id,
            "name": f"wide_sql_{case_id:04d}_{pct_name}_{gap_name}_s{int(score_min*100)}_amt{amount_min}_top{topn}_h{hold}_{sort_name}",
            "annual": annual,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "total_return": total,
            "active_days": active_days,
            "daily_rows": len(daily),
            "score_min": score_min,
            "pct_min": pct_min,
            "pct_max": pct_max,
            "buy_gap_min": gap_min,
            "buy_gap_max": gap_max,
            "amount_min": amount_min,
            "topn": topn,
            "hold": hold,
            "sort_mode": sort_name,
            "target_each": target_each,
        }
        rows.append(row)
        if annual > BASELINE_ANNUAL and mdd < 0.40:
            candidate_rows.append(row)
        case_id += 1

    out = pd.DataFrame(rows).sort_values(["annual", "sharpe"], ascending=False)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    manifest = pd.DataFrame(candidate_rows).sort_values(["annual", "sharpe"], ascending=False).head(10)
    signal_paths = []
    for _, r in manifest.iterrows():
        name = r["name"]
        ret_col = f"ret_h{int(r['hold'])}"
        order_expr = sort_modes[r["sort_mode"]]
        filters = [
            f"score_pct_rank >= {r['score_min']}",
            f"signal_amount >= {r['amount_min']}",
            f"buy_amount >= {r['amount_min']}",
            f"{ret_col} is not null",
        ]
        if not pd.isna(r["pct_min"]):
            filters.append(f"signal_pct_chg >= {r['pct_min']}")
        if not pd.isna(r["pct_max"]):
            filters.append(f"signal_pct_chg <= {r['pct_max']}")
        if not pd.isna(r["buy_gap_min"]):
            filters.append(f"buy_open_gap_raw_pct >= {r['buy_gap_min']}")
        if not pd.isna(r["buy_gap_max"]):
            filters.append(f"buy_open_gap_raw_pct <= {r['buy_gap_max']}")
        where = " and ".join(filters)
        sig = con.execute(
            f"""
            with picked as (
                select
                    *,
                    row_number() over(partition by buy_date order by {order_expr}) as pick_rank
                from active_l4_wide
                where {where}
            )
            select
                signal_date, buy_date, stock_code, name, market,
                pick_rank as rank,
                {float(r['target_each'])} as target_weight,
                pred_prob as signal_score,
                pred_prob, score_pct_rank,
                signal_pct_chg, signal_open_gap_raw_pct, buy_open_gap_raw_pct,
                signal_amount, buy_amount, signal_total_mv, buy_total_mv,
                {int(r['hold'])} as hold_days
            from picked
            where pick_rank <= {int(r['topn'])}
            order by buy_date, pick_rank
            """
        ).fetchdf()
        path = VARIANT_DIR / f"{name}.csv"
        sig.to_csv(path, index=False, encoding="utf-8-sig")
        signal_paths.append(str(path))
    if len(manifest):
        manifest = manifest.copy()
        manifest["signal_file"] = signal_paths
    manifest_path = REPORT_DIR / "active_l4_wide_sql_stage1_variant_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(out.head(30).to_string(index=False))
    print("summary", OUT_CSV)
    print("manifest", manifest_path, "rows", len(manifest))
    con.close()


if __name__ == "__main__":
    main()

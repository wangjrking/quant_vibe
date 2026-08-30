from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as attribution
import research_v260_fixed10_drawdown_position_attribution_20260822 as positions
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_industry_concentration_diagnostic_20260822"
)
L2_PATH = REPO / "quant/data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"


def daily_concentration(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"buy_date", "stock_code", "industry"}
    if not required.issubset(rows.columns):
        raise ValueError("industry concentration input schema mismatch")
    output = []
    for buy_date, group in rows.groupby("buy_date", sort=True):
        industries = group["industry"].astype(str).tolist()
        counts = Counter(industries)
        top_industry, top_count = max(
            counts.items(), key=lambda item: (item[1], item[0])
        )
        output.append(
            {
                "buy_date": str(buy_date),
                "position_count": int(len(group)),
                "industry_count": int(len(counts)),
                "max_industry_positions": int(top_count),
                "max_industry_share": float(top_count / len(group)),
                "top_industry": str(top_industry),
            }
        )
    return pd.DataFrame(output)


def summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("empty concentration frame")
    maxima = frame["max_industry_positions"].astype(int)
    return {
        "days": int(len(frame)),
        "mean_industry_count": float(frame["industry_count"].mean()),
        "mean_max_industry_positions": float(maxima.mean()),
        "p95_max_industry_positions": float(maxima.quantile(0.95)),
        "maximum_industry_positions": int(maxima.max()),
        "days_max_industry_at_least_3": int((maxima >= 3).sum()),
        "days_max_industry_at_least_4": int((maxima >= 4).sum()),
        "days_max_industry_at_least_5": int((maxima >= 5).sum()),
        "ratio_max_industry_at_least_3": float((maxima >= 3).mean()),
        "ratio_max_industry_at_least_4": float((maxima >= 4).mean()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily, actions, observations, context = positions.run_with_observer()
    episode = attribution.maximum_drawdown_episode(daily)
    pairs = pd.DataFrame(
        [
            {"buy_date": item["buy_date"], "stock_code": stock_code}
            for item in observations
            for stock_code in item["positions_after_trades"]
        ]
    )
    if pairs.duplicated(["buy_date", "stock_code"]).any():
        raise RuntimeError("duplicate held-position key")
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{L2_PATH.as_posix()}' AS l2 (READ_ONLY)")
        con.register("required_pairs", pairs)
        resolved = con.execute(
            """
            SELECT p.buy_date, p.stock_code,
                   latest.industry,
                   latest.source_trade_date
            FROM required_pairs p
            LEFT JOIN LATERAL (
              SELECT d.industry, d.trade_date AS source_trade_date
              FROM l2.STOCK_DAILY_DATA d
              WHERE d.stock_code=p.stock_code
                AND d.trade_date <= p.buy_date
                AND d.industry IS NOT NULL
                AND length(trim(CAST(d.industry AS VARCHAR))) > 0
              ORDER BY d.trade_date DESC
              LIMIT 1
            ) latest ON TRUE
            ORDER BY p.buy_date, p.stock_code
            """
        ).fetchdf()
    finally:
        con.close()
    if len(resolved) != len(pairs):
        raise RuntimeError("held-position industry join changed row count")
    missing = resolved["industry"].isna() | (resolved["industry"].astype(str).str.len() == 0)
    if missing.any():
        raise RuntimeError(f"held-position industry missing: {int(missing.sum())}")
    future_source = (
        resolved["source_trade_date"].astype(str) > resolved["buy_date"].astype(str)
    )
    if future_source.any():
        raise RuntimeError("future industry label entered concentration diagnostic")

    concentration = daily_concentration(resolved)
    full_summary = summary(concentration)
    drawdown_frame = concentration[
        (concentration["buy_date"] >= episode["peak_date"])
        & (concentration["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    result = {
        "status": "diagnostic_complete_2026_not_opened",
        "source_strategy": context["rules"]["strategy_id"],
        "development_boundary": [research_base.FIRST_BUY, round1.DEVELOPMENT_END],
        "held_position_pairs": int(len(pairs)),
        "missing_industry_pairs": int(missing.sum()),
        "prior_industry_label_pairs": int(
            (
                resolved["source_trade_date"].astype(str)
                < resolved["buy_date"].astype(str)
            ).sum()
        ),
        "future_industry_label_pairs": int(future_source.sum()),
        "all_period": full_summary,
        "maximum_drawdown_episode": episode,
        "maximum_drawdown_industry_concentration": summary(drawdown_frame),
        "worst_concentration_days": concentration.sort_values(
            ["max_industry_positions", "buy_date"], ascending=[False, True]
        )
        .head(20)
        .to_dict("records"),
        "data_access": context["access"],
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "industry_concentration_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

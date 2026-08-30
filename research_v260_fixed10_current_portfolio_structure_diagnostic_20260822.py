from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_drawdown_attribution_20260822 as drawdown
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_industry_concentration_diagnostic_20260822 as industry
import research_v260_fixed10_position_correlation_diagnostic_20260822 as correlation
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_style_exposure_diagnostic_20260822 as style
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as current_run
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_current_portfolio_structure_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def correlation_frame(observations: list[dict], arrays: dict) -> pd.DataFrame:
    stock_index = {str(code): idx for idx, code in enumerate(arrays["stocks"])}
    date_index = {str(date): idx for idx, date in enumerate(arrays["dates"])}
    returns = correlation.returns_from_close(arrays["close_qfq"])
    rows = []
    for observation in observations:
        day = date_index[str(observation["buy_date"])]
        indices = [
            stock_index[str(code)] for code in observation["positions_after_trades"]
        ]
        values = correlation.pairwise_correlations(
            returns,
            day,
            indices,
            correlation.LOOKBACK,
            correlation.MIN_OBSERVATIONS,
        )
        if values:
            rows.append(
                {
                    "buy_date": str(observation["buy_date"]),
                    "pair_count": int(len(values)),
                    "mean_pairwise_correlation": float(np.mean(values)),
                    "max_pairwise_correlation": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def industry_frame(observations: list[dict]) -> tuple[pd.DataFrame, dict]:
    pairs = pd.DataFrame(
        [
            {"buy_date": str(item["buy_date"]), "stock_code": str(code)}
            for item in observations
            for code in item["positions_after_trades"]
        ]
    )
    if pairs.duplicated(["buy_date", "stock_code"]).any():
        raise RuntimeError("duplicate current held-position key")
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{industry.L2_PATH.as_posix()}' AS l2 (READ_ONLY)")
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
        raise RuntimeError("current industry join changed row count")
    missing = resolved["industry"].isna()
    future = resolved["source_trade_date"].astype(str) > resolved["buy_date"].astype(str)
    if missing.any() or future.any():
        raise RuntimeError("current industry contract failed")
    return industry.daily_concentration(resolved), {
        "held_position_pairs": int(len(pairs)),
        "missing_industry_pairs": int(missing.sum()),
        "future_industry_pairs": int(future.sum()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered current portfolio structure diagnostic")
    context = harness.load_context(checkpoint["selected_policy"])
    daily, actions, observations, _, _, _ = current_run.run_with_observer(
        context, checkpoint["selected_policy"]
    )
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    equivalence = {
        key: bool(
            np.isclose(
                metrics[key],
                checkpoint["current_best_equalweight"][key],
                rtol=0.0,
                atol=1e-12,
            )
        )
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover_annualized",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("current portfolio structure baseline drifted")

    episode = drawdown.maximum_drawdown_episode(daily)
    corr = correlation_frame(observations, context.arrays)
    corr_drawdown = corr[
        (corr["buy_date"] >= episode["peak_date"])
        & (corr["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    exposure = style.build_daily_exposure(observations, context.arrays)
    exposure_drawdown = exposure[
        (exposure["buy_date"] >= episode["peak_date"])
        & (exposure["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)
    concentration, industry_contract = industry_frame(observations)
    concentration_drawdown = concentration[
        (concentration["buy_date"] >= episode["peak_date"])
        & (concentration["buy_date"] <= episode["trough_date"])
    ].reset_index(drop=True)

    all_style = style.summarize(exposure)
    drawdown_style = style.summarize(exposure_drawdown)
    all_corr = correlation.summarize(corr)
    drawdown_corr = correlation.summarize(corr_drawdown)
    all_industry = industry.summary(concentration)
    drawdown_industry = industry.summary(concentration_drawdown)
    result = {
        "status": "current_portfolio_structure_diagnostic_complete_2026_not_opened",
        "maximum_drawdown_episode": episode,
        "correlation": {
            "all_period": all_corr,
            "drawdown_episode": drawdown_corr,
            "drawdown_minus_all_mean": float(
                drawdown_corr["mean_pairwise_correlation"]
                - all_corr["mean_pairwise_correlation"]
            ),
        },
        "style": {
            "all_period": all_style,
            "drawdown_episode": drawdown_style,
            "drawdown_minus_all_mean_percentile": {
                field: float(
                    drawdown_style[field]["mean_percentile"]
                    - all_style[field]["mean_percentile"]
                )
                for field in style.STYLE_FIELDS
            },
        },
        "industry": {
            "all_period": all_industry,
            "drawdown_episode": drawdown_industry,
            "contract": industry_contract,
        },
        "structural_flags": {
            "drawdown_correlation_materially_higher": bool(
                drawdown_corr["mean_pairwise_correlation"]
                > all_corr["mean_pairwise_correlation"] + 0.10
            ),
            "drawdown_size_percentile_materially_lower": bool(
                drawdown_style["total_mv"]["mean_percentile"]
                < all_style["total_mv"]["mean_percentile"] - 0.10
            ),
            "drawdown_industry_concentration_materially_higher": bool(
                drawdown_industry["mean_max_industry_positions"]
                > all_industry["mean_max_industry_positions"] + 0.50
            ),
        },
        "baseline_equivalence": equivalence,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "portfolio_structure.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

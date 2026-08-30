from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_weak2_drawdown_position_attribution_20260822 as current_run
import research_v260_lowrisk_score_tuning_20260821 as research_base


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_replacement_counterfactual_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
HORIZONS = (5, 10, 20)


def forward_return(close: np.ndarray, start: int, idx: int, horizon: int) -> float:
    end = start + int(horizon)
    if end >= close.shape[0]:
        return np.nan
    first, last = float(close[start, idx]), float(close[end, idx])
    if not np.isfinite(first) or not np.isfinite(last) or first <= 0.0:
        return np.nan
    return last / first - 1.0


def summarize(rows: list[dict]) -> dict:
    result = {"transition_days": int(len(rows))}
    for horizon in HORIZONS:
        key = f"replacement_excess_{horizon}d"
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[f"observed_{horizon}d"] = int(len(values))
        result[f"mean_replacement_excess_{horizon}d"] = (
            float(np.mean(values)) if len(values) else None
        )
        result[f"median_replacement_excess_{horizon}d"] = (
            float(np.median(values)) if len(values) else None
        )
        result[f"replacement_win_rate_{horizon}d"] = (
            float(np.mean(values > 0.0)) if len(values) else None
        )
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered replacement diagnostic")
    context = harness.load_context(checkpoint["selected_policy"])
    daily, actions, observations, _, _, _ = current_run.run_with_observer(
        context, checkpoint["selected_policy"]
    )
    expected = checkpoint["current_best_equalweight"]
    metrics = round1.evaluate_run(
        daily, actions, research_base.FIRST_BUY, round1.DEVELOPMENT_END
    )
    equivalence = {
        key: bool(np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12))
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    }
    if not all(equivalence.values()):
        raise RuntimeError("replacement diagnostic baseline drifted")

    stocks = np.asarray(context.arrays["stocks"], dtype=str)
    dates = np.asarray(context.arrays["dates"], dtype=str)
    stock_index = {code: idx for idx, code in enumerate(stocks)}
    date_index = {date: idx for idx, date in enumerate(dates)}
    close = np.asarray(context.arrays["close_qfq"], dtype=np.float64)
    rows = []
    previous: set[str] = set()
    for item in observations:
        current = set(str(code) for code in item["positions_after_trades"])
        exited = sorted(previous - current)
        entered = sorted(current - previous)
        previous = current
        if not exited or not entered:
            continue
        signal_date = str(item["signal_date"])
        buy_date = str(item["buy_date"])
        signal_idx = date_index[signal_date]
        buy_idx = date_index[buy_date]
        exit_scores = np.asarray(
            [context.score[signal_idx, stock_index[code]] for code in exited],
            dtype=np.float64,
        )
        entry_scores = np.asarray(
            [context.score[signal_idx, stock_index[code]] for code in entered],
            dtype=np.float64,
        )
        exit_type = (
            "score_exit"
            if np.all(np.isfinite(exit_scores) & (exit_scores < 0.85))
            else "mandatory_or_mixed_exit"
        )
        row = {
            "signal_date": signal_date,
            "buy_date": buy_date,
            "exit_type": exit_type,
            "exited_count": int(len(exited)),
            "entered_count": int(len(entered)),
            "mean_exit_score": float(np.nanmean(exit_scores)),
            "mean_entry_score": float(np.nanmean(entry_scores)),
            "mean_score_advantage": float(
                np.nanmean(entry_scores) - np.nanmean(exit_scores)
            ),
        }
        for horizon in HORIZONS:
            sold_returns = np.asarray(
                [
                    forward_return(close, buy_idx, stock_index[code], horizon)
                    for code in exited
                ],
                dtype=np.float64,
            )
            bought_returns = np.asarray(
                [
                    forward_return(close, buy_idx, stock_index[code], horizon)
                    for code in entered
                ],
                dtype=np.float64,
            )
            sold = sold_returns[np.isfinite(sold_returns)]
            bought = bought_returns[np.isfinite(bought_returns)]
            row[f"replacement_excess_{horizon}d"] = (
                float(np.mean(bought) - np.mean(sold))
                if len(sold) and len(bought)
                else np.nan
            )
        rows.append(row)

    by_type = {
        exit_type: summarize([row for row in rows if row["exit_type"] == exit_type])
        for exit_type in sorted({row["exit_type"] for row in rows})
    }
    by_year = {
        year: summarize([row for row in rows if row["buy_date"].startswith(year)])
        for year in sorted({row["buy_date"][:4] for row in rows})
    }
    advantage_bands = {
        "below_010": lambda value: value < 0.10,
        "010_to_020": lambda value: 0.10 <= value < 0.20,
        "at_least_020": lambda value: value >= 0.20,
    }
    by_score_advantage = {
        name: summarize(
            [row for row in rows if predicate(row["mean_score_advantage"])]
        )
        for name, predicate in advantage_bands.items()
    }
    entry_score_bands = {
        "below_090": lambda value: value < 0.90,
        "090_to_095": lambda value: 0.90 <= value < 0.95,
        "at_least_095": lambda value: value >= 0.95,
    }
    by_entry_score = {
        name: summarize([row for row in rows if predicate(row["mean_entry_score"])])
        for name, predicate in entry_score_bands.items()
    }
    result = {
        "status": "replacement_counterfactual_diagnostic_complete_2026_not_opened",
        "method": (
            "for each actual position replacement day, compare subsequent qfq-close "
            "returns of newly entered names against exited names; diagnostics do not "
            "alter the strategy"
        ),
        "overall": summarize(rows),
        "by_exit_type": by_type,
        "by_year": by_year,
        "by_score_advantage": by_score_advantage,
        "by_entry_score": by_entry_score,
        "baseline_equivalence": equivalence,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "replacement_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

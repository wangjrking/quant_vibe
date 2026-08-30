from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_pressure_recheck_harness_20260822 as harness
import research_v260_fixed10_rolling_relative_robustness_20260822 as rolling


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_action_rank_diagnostic_20260822"
)
CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_confirmed_weak2_vol_priority_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)


def summary(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def enrich_actions(context, actions: pd.DataFrame) -> pd.DataFrame:
    date_index = {
        str(value): index for index, value in enumerate(context.arrays["dates"])
    }
    stock_index = {
        str(value): index for index, value in enumerate(context.arrays["stocks"])
    }
    rows = []
    for row in actions.itertuples(index=False):
        date = str(row.signal_date)
        code = str(row.stock_code)
        t = date_index.get(date)
        s = stock_index.get(code)
        if t is None or s is None:
            raise RuntimeError(f"action key not found in score matrix: {date}/{code}")
        ranked = context.order[t]
        position = np.flatnonzero(ranked == s)
        if position.size != 1:
            raise RuntimeError(f"action rank is not unique: {date}/{code}")
        score = float(context.score[t, s])
        if not np.isfinite(score):
            raise RuntimeError(f"action score is non-finite: {date}/{code}")
        rows.append({
            "signal_date": date,
            "buy_date": str(row.buy_date),
            "action": str(row.action),
            "stock_code": code,
            "score": score,
            "rank_1based": int(position[0]) + 1,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint["validation_2026_opened"]:
        raise PermissionError("2026 data entered action-rank diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = harness.load_context(policy)
    daily, actions = rolling.run_current(context, policy)
    metrics = round1.evaluate_run(
        daily, actions, "20220607", round1.DEVELOPMENT_END
    )
    expected = checkpoint["current_best_equalweight"]
    equivalent = all(
        np.isclose(metrics[key], expected[key], rtol=0.0, atol=1e-12)
        for key in (
            "cumulative_return", "cagr", "sharpe", "max_drawdown",
            "turnover_annualized", "average_invested_ratio",
        )
    )
    if not equivalent:
        raise RuntimeError("action-rank baseline drifted")

    enriched = enrich_actions(context, actions)
    buys = enriched.loc[enriched["action"] == "BUY"].copy()
    sells = enriched.loc[enriched["action"] == "SELL"].copy()
    initial_date = str(buys["signal_date"].min())
    initial = buys.loc[buys["signal_date"] == initial_date]
    replacements = buys.loc[buys["signal_date"] != initial_date]
    replacement_ranks = replacements["rank_1based"].to_numpy(dtype=np.float64)
    sell_ranks = sells["rank_1based"].to_numpy(dtype=np.float64)

    result = {
        "status": "action_rank_diagnostic_complete_2026_not_opened",
        "initial_fill": {
            "signal_date": initial_date,
            "rank_summary": summary(initial["rank_1based"].to_numpy(dtype=np.float64)),
        },
        "replacement_buys": {
            "rank_summary": summary(replacement_ranks),
            "fraction_rank_le_10": float(np.mean(replacement_ranks <= 10)),
            "fraction_rank_le_20": float(np.mean(replacement_ranks <= 20)),
            "fraction_rank_le_50": float(np.mean(replacement_ranks <= 50)),
            "fraction_rank_le_100": float(np.mean(replacement_ranks <= 100)),
            "score_summary": summary(
                replacements["score"].to_numpy(dtype=np.float64)
            ),
        },
        "sells": {
            "rank_summary": summary(sell_ranks),
            "score_summary": summary(sells["score"].to_numpy(dtype=np.float64)),
            "fraction_score_below_085": float(np.mean(sells["score"] < 0.85)),
        },
        "baseline_equivalent": equivalent,
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "action_rank_diagnostic.json", result)
    enriched.to_csv(OUTPUT_ROOT / "action_rank_rows.csv", index=False)
    print(json.dumps({
        "status": result["status"],
        "replacement_buys": result["replacement_buys"],
        "sells": result["sells"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

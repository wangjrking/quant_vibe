from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_daily_exit_after_renewal_gate_20260822 as final_round
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_lowrisk_score_tuning_20260821 as research_base
from research_v260_runtime import fixed10_renewal_quality_v109 as renewal_quality_v109


OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/strategy_agent_v260_fixed10_drawdown_attribution_20260822"
)


def maximum_drawdown_episode(daily: pd.DataFrame) -> dict:
    equity = daily["equity"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    trough_index = int(np.argmin(drawdowns))
    peak_index = int(np.argmax(equity[: trough_index + 1]))
    return {
        "peak_date": str(daily.iloc[peak_index]["date"]),
        "trough_date": str(daily.iloc[trough_index]["date"]),
        "drawdown": float(-drawdowns[trough_index]),
        "peak_equity": float(equity[peak_index]),
        "trough_equity": float(equity[trough_index]),
        "trading_days": int(trough_index - peak_index),
    }


def worst_compound_window(daily: pd.DataFrame, window: int) -> dict:
    returns = daily["return"].to_numpy(dtype=float)
    compounded = (
        pd.Series(np.log1p(returns)).rolling(window).sum().pipe(np.expm1)
    )
    end_index = int(compounded.idxmin())
    start_index = end_index - window + 1
    return {
        "window": int(window),
        "start_date": str(daily.iloc[start_index]["date"]),
        "end_date": str(daily.iloc[end_index]["date"]),
        "return": float(compounded.iloc[end_index]),
    }


def relative_window(candidate: pd.DataFrame, production: pd.DataFrame, window: int) -> dict:
    if not candidate["date"].astype(str).equals(production["date"].astype(str)):
        raise RuntimeError("candidate and production dates differ")
    relative_log_return = np.log1p(candidate["return"].to_numpy(dtype=float)) - np.log1p(
        production["return"].to_numpy(dtype=float)
    )
    compounded = pd.Series(relative_log_return).rolling(window).sum().pipe(np.expm1)
    end_index = int(compounded.idxmin())
    start_index = end_index - window + 1
    return {
        "window": int(window),
        "start_date": str(candidate.iloc[start_index]["date"]),
        "end_date": str(candidate.iloc[end_index]["date"]),
        "candidate_minus_production_return": float(compounded.iloc[end_index]),
    }


def actions_in_window(actions: pd.DataFrame, start: str, end: str) -> dict:
    dates = actions["buy_date"].astype(str)
    selected = actions[(dates >= start) & (dates <= end)]
    return {
        "buy_count": int((selected["action"] == "BUY").sum()),
        "sell_count": int((selected["action"] == "SELL").sum()),
        "unique_stocks": int(selected["stock_code"].nunique()),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    harness, protocol, rules, manifests, arrays, access = round1.load_arrays(
        round1.DEVELOPMENT_END
    )
    if access["logical_max_date"] != round1.DEVELOPMENT_END:
        raise PermissionError("2026 data entered drawdown attribution")
    definition = harness.production_definition(protocol)
    score, order = harness.v95.score_pair(arrays, 0.0, 7, 0.1)
    production_daily, production_actions = research_base.run_shell(
        harness,
        arrays,
        protocol,
        score,
        order,
        definition,
        "production_shell",
        round1.DEVELOPMENT_END,
        actions=True,
    )
    candidate_daily, candidate_actions = round1.run_fixed10(
        harness,
        arrays,
        protocol,
        definition,
        score,
        order,
        final_round.candidate_policy(1),
        round1.DEVELOPMENT_END,
        slip=round1.BASELINE_COST,
        record_actions=True,
        simulator=renewal_quality_v109.simulate,
    )
    candidate_episode = maximum_drawdown_episode(candidate_daily)
    production_episode = maximum_drawdown_episode(production_daily)
    result = {
        "status": "pre2026_diagnostic_2026_not_opened",
        "source_strategy": rules["strategy_id"],
        "candidate_policy": final_round.candidate_policy(1),
        "production_maximum_drawdown": production_episode,
        "candidate_maximum_drawdown": candidate_episode,
        "candidate_actions_in_maximum_drawdown": actions_in_window(
            candidate_actions,
            candidate_episode["peak_date"],
            candidate_episode["trough_date"],
        ),
        "production_actions_in_maximum_drawdown": actions_in_window(
            production_actions,
            production_episode["peak_date"],
            production_episode["trough_date"],
        ),
        "candidate_worst_windows": {
            str(window): worst_compound_window(candidate_daily, window)
            for window in (20, 60)
        },
        "production_worst_windows": {
            str(window): worst_compound_window(production_daily, window)
            for window in (20, 60)
        },
        "worst_relative_windows": {
            str(window): relative_window(candidate_daily, production_daily, window)
            for window in (20, 60)
        },
        "daily_hashes": {
            "production": round1.frame_hash(production_daily),
            "candidate": round1.frame_hash(candidate_daily),
        },
        "data_access": access,
        "source_manifests": manifests,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    round1.atomic_json(OUTPUT_ROOT / "drawdown_attribution.json", result)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

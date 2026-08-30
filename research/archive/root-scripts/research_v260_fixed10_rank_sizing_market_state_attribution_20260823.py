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

import research_v260_fixed10_confirmed_severe_weak_defensive_sleeve_20260822 as confirmed
import research_v260_fixed10_entry_rank_sizing_profit_optimization_20260822 as sizing
import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_regime_exit_threshold_20260822 as regime


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_market_state_attribution_20260823"
)


def state_summary(frame: pd.DataFrame) -> dict:
    candidate = frame["candidate_return"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_return"].to_numpy(dtype=np.float64)
    log_excess = np.log1p(candidate) - np.log1p(baseline)
    return {
        "days": int(len(frame)),
        "candidate_cumulative_return": float(np.prod(1.0 + candidate) - 1.0),
        "equalweight_cumulative_return": float(np.prod(1.0 + baseline) - 1.0),
        "candidate_minus_equalweight_log_excess": float(log_excess.sum()),
        "candidate_minus_equalweight_compounded_excess": float(
            np.exp(log_excess.sum()) - 1.0
        ),
        "candidate_outperforms_day_fraction": float((log_excess > 0.0).mean()),
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered market-state attribution")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = sizing.width_tools.harness.load_context(policy)
    equal_daily, equal_actions = sizing.run_case(
        context, policy, sizing.BASELINE_COST, None, None
    )
    rank_daily, rank_actions = sizing.run_case(
        context,
        policy,
        sizing.BASELINE_COST,
        sizing.TOP_WEIGHT_MULTIPLIER,
        sizing.BOTTOM_WEIGHT_MULTIPLIER,
    )
    if not equal_daily["date"].astype(str).equals(rank_daily["date"].astype(str)):
        raise RuntimeError("market-state attribution dates do not align")
    strong = regime.strong_market_mask(context.score, context.protocol)
    confirmed_weak = confirmed.confirmed_weak_mask(strong, 2)
    state_by_date = {
        str(date): (
            "strong"
            if bool(strong[index])
            else "confirmed_weak"
            if bool(confirmed_weak[index])
            else "weak_transition"
        )
        for index, date in enumerate(context.arrays["dates"])
    }
    frame = pd.DataFrame(
        {
            "date": equal_daily["date"].astype(str),
            "baseline_return": equal_daily["return"].to_numpy(dtype=np.float64),
            "candidate_return": rank_daily["return"].to_numpy(dtype=np.float64),
        }
    )
    frame["market_state"] = frame["date"].map(state_by_date)
    if frame["market_state"].isna().any():
        raise RuntimeError("market state is missing for a development date")
    states = {
        state: state_summary(group)
        for state, group in frame.groupby("market_state", sort=True)
    }
    overall = state_summary(frame)
    output = {
        "status": "rank_sizing_market_state_attribution_complete_2026_not_opened",
        "role": "diagnostic_only_no_new_rule_or_gate",
        "market_state_semantics": {
            "strong": "existing frozen strong-market mask",
            "confirmed_weak": "existing frozen weak state confirmed for two sessions",
            "weak_transition": "weak state before two-session confirmation",
        },
        "overall": overall,
        "states": states,
        "state_log_excess_reconciles": bool(
            np.isclose(
                sum(
                    item["candidate_minus_equalweight_log_excess"]
                    for item in states.values()
                ),
                overall["candidate_minus_equalweight_log_excess"],
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "deterministic_hashes": {
            "equal_daily": round1.frame_hash(equal_daily),
            "equal_actions": round1.frame_hash(equal_actions),
            "rank_daily": round1.frame_hash(rank_daily),
            "rank_actions": round1.frame_hash(rank_actions),
        },
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "market_state_attribution.json", output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()

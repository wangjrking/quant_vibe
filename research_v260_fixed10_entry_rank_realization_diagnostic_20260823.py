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
import research_v260_fixed10_rank_sizing_mechanism_attribution_20260823 as mechanism


CHECKPOINT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_rank_sizing_checkpoint_20260822/"
    "pre2026_checkpoint.json"
)
OUTPUT_ROOT = (
    REPO
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed10_entry_rank_realization_diagnostic_20260823"
)
TARGET_TOLERANCE = 1e-10


def implied_entry_rank(target_pct: float, positions: int = 10) -> int | None:
    value = float(target_pct)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("entry target must be finite and positive")
    schedule = np.linspace(1.10, 0.90, positions, dtype=np.float64) / positions
    matches = np.flatnonzero(np.isclose(schedule, value, rtol=0.0, atol=TARGET_TOLERANCE))
    if len(matches) > 1:
        raise RuntimeError("entry target maps to multiple ranks")
    return int(matches[0] + 1) if len(matches) == 1 else None


def completed_position_lifecycles(
    observer_records: list[dict], actions: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    required = {"buy_date", "action", "stock_code", "target_pct"}
    if not required.issubset(actions.columns):
        raise ValueError("action frame is missing lifecycle columns")
    actions_by_date = {
        str(date): group.reset_index(drop=True)
        for date, group in actions.groupby("buy_date", sort=False)
    }
    active: dict[str, dict] = {}
    completed = []
    for record in observer_records:
        date = str(record["buy_date"])
        for mark in record["mark_records"]:
            code = str(mark["stock_code"])
            if code not in active:
                raise RuntimeError(f"held position has no active lifecycle: {date}/{code}")
            prior_notional = float(mark["shares_before"]) * float(mark["prior_price"])
            if not np.isfinite(prior_notional) or prior_notional <= 0.0:
                raise ValueError("prior notional must be finite and positive")
            mark_return = float(mark["mark_pnl"]) / prior_notional
            if not np.isfinite(mark_return) or mark_return <= -1.0:
                raise ValueError("mark return is invalid")
            active[code]["growth"] *= 1.0 + mark_return
            active[code]["holding_days"] += 1

        for action in actions_by_date.get(date, pd.DataFrame()).itertuples(index=False):
            code = str(action.stock_code)
            side = str(action.action)
            if side == "SELL":
                if float(action.target_pct) <= TARGET_TOLERANCE and code in active:
                    item = active.pop(code)
                    item["exit_date"] = date
                    item["lifecycle_return"] = float(item.pop("growth") - 1.0)
                    completed.append(item)
            elif side == "BUY":
                if code not in active:
                    target_pct = float(action.target_pct)
                    active[code] = {
                        "stock_code": code,
                        "entry_date": date,
                        "entry_year": date[:4],
                        "target_pct": target_pct,
                        "entry_rank": implied_entry_rank(target_pct),
                        "growth": 1.0,
                        "holding_days": 0,
                    }
            else:
                raise ValueError(f"unsupported action: {side}")
    frame = pd.DataFrame(completed)
    return frame, len(active)


def summarize_lifecycles(frame: pd.DataFrame, open_positions: int) -> dict:
    if frame.empty:
        raise ValueError("no completed position lifecycles")
    ranked = frame.loc[frame["entry_rank"].notna()].copy()
    ranked["entry_rank"] = ranked["entry_rank"].astype(int)
    correlation = float(
        ranked["entry_rank"].corr(ranked["lifecycle_return"], method="spearman")
    )
    groups = {}
    for rank, group in ranked.groupby("entry_rank", sort=True):
        values = group["lifecycle_return"].to_numpy(dtype=np.float64)
        groups[str(int(rank))] = {
            "completed_positions": int(len(values)),
            "mean_lifecycle_return": float(values.mean()),
            "median_lifecycle_return": float(np.median(values)),
            "positive_fraction": float((values > 0.0).mean()),
            "mean_holding_days": float(group["holding_days"].mean()),
        }
    return {
        "completed_positions": int(len(frame)),
        "ranked_completed_positions": int(len(ranked)),
        "refill_or_equalweight_completed_positions": int(
            frame["entry_rank"].isna().sum()
        ),
        "open_positions_excluded": int(open_positions),
        "entry_rank_vs_lifecycle_return_spearman": correlation,
        "expected_direction_if_higher_score_is_better": "negative",
        "rank_groups": groups,
        "year_groups": {
            str(year): {
                "completed_positions": int(len(group)),
                "rank_return_spearman": (
                    float(
                        group.loc[group["entry_rank"].notna(), "entry_rank"].corr(
                            group.loc[
                                group["entry_rank"].notna(), "lifecycle_return"
                            ],
                            method="spearman",
                        )
                    )
                    if group["entry_rank"].notna().sum() >= 2
                    else None
                ),
            }
            for year, group in frame.groupby("entry_year", sort=True)
        },
    }


def main() -> None:
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("validation_2026_opened") is not False:
        raise PermissionError("2026 entered entry-rank realization diagnostic")
    policy = copy.deepcopy(checkpoint["selected_policy"])
    context = mechanism.sizing.width_tools.harness.load_context(policy)
    daily, actions, records = mechanism.run_with_observer(context, policy, True)
    metrics = mechanism.sizing.evaluate(daily, actions)
    for key in ("cumulative_return", "cagr", "sharpe", "max_drawdown"):
        if not np.isclose(
            metrics[key], checkpoint["metrics_0_30pct"][key], rtol=0.0, atol=1e-12
        ):
            raise RuntimeError("entry-rank diagnostic replay drifted")
    lifecycles, open_positions = completed_position_lifecycles(records, actions)
    summary = summarize_lifecycles(lifecycles, open_positions)
    result = {
        "status": "entry_rank_realization_diagnostic_complete_2026_not_opened",
        "role": "economic_mechanism_diagnostic_not_a_selection_gate",
        "candidate_id": checkpoint["selected_candidate"],
        "summary": summary,
        "interpretation": (
            "negative rank-return correlation supports allocating more to higher "
            "score ranks; weak or unstable correlation is disclosed but does not "
            "create a new rule or rejection gate"
        ),
        "data_access": context.access,
        "validation_2026_opened": False,
        "production_modified": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "entry_rank_realization.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_positions": summary["completed_positions"],
                "rank_return_spearman": summary[
                    "entry_rank_vs_lifecycle_return_spearman"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

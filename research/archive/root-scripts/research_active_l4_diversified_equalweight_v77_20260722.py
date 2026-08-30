# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_fast_health_neighborhood_v75_20260722 as health_lib
import research_active_l4_independent_sell_v36_20260721 as sell_lib
import research_active_l4_lagged_health_liquidity_v66_20260721 as quality_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_diversified_equalweight_v77_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class PortfolioProfile:
    top_n: int
    gross_exposure: float
    amount_min: int
    mv_min: int
    turnover_max: float


def stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_liquidity(state, arrays, profile: PortfolioProfile):
    score, order, masked, exposure = state
    masked = {key: value.copy() for key, value in masked.items()}
    valid = np.isfinite(arrays["turnover_rate"])
    valid &= arrays["turnover_rate"] >= 0.0
    valid &= arrays["turnover_rate"] <= profile.turnover_max
    valid &= arrays["amount"] >= profile.amount_min
    valid &= arrays["total_mv"] >= profile.mv_min
    masked["signal_clean"] &= valid
    return score, order, masked, exposure


def simulate(
    arrays: dict[str, np.ndarray],
    state,
    portfolio: PortfolioProfile,
    exit_profile: sell_lib.ExitProfile,
    end_date: str,
    *,
    record_actions: bool = False,
):
    score, order, masked, exposure = state
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    board_rate = core.board_limit_rate(stocks)
    sell_score = sell_lib.exit_score_array(arrays, score, exit_profile.exit_score_source)
    cash = 700_000.0
    previous_equity = 700_000.0
    shares: dict[int, float] = {}
    entry_index: dict[int, int] = {}
    last_price: dict[int, float] = {}
    rows: list[dict] = []
    actions: list[dict] = []

    for t, signal_date in enumerate(dates[:-1]):
        if signal_date > end_date:
            break
        buy_date = str(dates[t + 1])
        opens = arrays["buy_open"][t]
        pre_close = arrays["buy_pre_close"][t]
        valid_open = (
            np.isfinite(opens)
            & (opens > 0)
            & np.isfinite(pre_close)
            & (pre_close > 0)
        )
        for idx in list(shares):
            if valid_open[idx]:
                last_price[idx] = float(opens[idx])
        equity_before = cash + sum(
            shares[idx] * last_price.get(idx, 0.0) for idx in shares
        )

        universe = masked["signal_clean"][t] & masked["buy_clean"][t]
        universe &= valid_open & np.isfinite(score[t])
        universe &= arrays["listed_days"][t] >= 60
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]

        target_positions = int(
            np.floor(portfolio.top_n * float(exposure[t]) + 1e-9)
        )
        target_positions = max(min(target_positions, portfolio.top_n), 0)
        best_unheld = max(
            (float(score[t, idx]) for idx in ranked if idx not in shares),
            default=-np.inf,
        )
        normal_sells: list[int] = []
        for idx in list(shares):
            age = t - entry_index[idx]
            held_exit = (
                float(sell_score[t, idx])
                if np.isfinite(sell_score[t, idx])
                else -np.inf
            )
            held_buy = (
                float(score[t, idx]) if np.isfinite(score[t, idx]) else -np.inf
            )
            score_exit = (
                age >= exit_profile.min_hold
                and held_exit < exit_profile.sell_rank_below
                and best_unheld - held_buy >= exit_profile.replacement_advantage
            )
            if age >= exit_profile.max_hold or score_exit:
                normal_sells.append(idx)

        remaining = [idx for idx in shares if idx not in normal_sells]
        excess = max(len(remaining) - target_positions, 0)
        scale_sells = sorted(
            (
                idx
                for idx in remaining
                if t - entry_index[idx] >= exit_profile.min_hold
            ),
            key=lambda idx: (
                float(sell_score[t, idx])
                if np.isfinite(sell_score[t, idx])
                else -np.inf,
                stocks[idx],
            ),
        )[:excess]

        turnover = 0.0
        trades = 0
        sell_failed = False
        for idx in normal_sells + scale_sells:
            if not valid_open[idx]:
                sell_failed = True
                continue
            rate = corrected.corrected_limit_rate(arrays, t, idx, board_rate)
            if opens[idx] <= pre_close[idx] * (1.0 - rate) * 1.005:
                sell_failed = True
                continue
            gross = float(shares[idx] * opens[idx])
            slip = core.adaptive_slippage(gross, arrays["amount"][t, idx], "sell")
            cash += gross * (1.0 - slip) * (1.0 - 0.0003 - 0.0005)
            turnover += gross
            trades += 1
            if record_actions:
                actions.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "action": "SELL",
                        "stock_code": stocks[idx],
                        "target_pct": 0.0,
                        "execution_open_raw": float(opens[idx]),
                    }
                )
            del shares[idx]
            del entry_index[idx]
            last_price.pop(idx, None)

        slots = 0 if sell_failed else max(target_positions - len(shares), 0)
        candidates = [idx for idx in ranked if idx not in shares][:slots]
        # Equal weight is defined against the full target portfolio, not only
        # the currently empty slots. This prevents a one-slot refill from
        # receiving the entire portfolio budget.
        target_pct = portfolio.gross_exposure / portfolio.top_n
        for idx in candidates:
            gross_budget = min(equity_before * target_pct, cash / 1.001)
            if gross_budget < 1000:
                continue
            slip = core.adaptive_slippage(
                gross_budget, arrays["amount"][t, idx], "buy"
            )
            buy_price = float(opens[idx]) * (1.0 + slip)
            quantity = gross_budget / (buy_price * (1.0 + 0.0003))
            spend = quantity * buy_price * (1.0 + 0.0003)
            if quantity <= 0 or spend > cash + 1e-6:
                continue
            cash -= spend
            shares[idx] = quantity
            entry_index[idx] = t
            last_price[idx] = float(opens[idx])
            turnover += quantity * float(opens[idx])
            trades += 1
            if record_actions:
                actions.append(
                    {
                        "signal_date": signal_date,
                        "buy_date": buy_date,
                        "action": "BUY",
                        "stock_code": stocks[idx],
                        "target_pct": target_pct,
                        "execution_open_raw": float(opens[idx]),
                    }
                )

        equity_after = cash + sum(
            shares[idx] * last_price.get(idx, 0.0) for idx in shares
        )
        rows.append(
            {
                "date": buy_date,
                "return": equity_after / previous_equity - 1.0,
                "equity": equity_after,
                "turnover": turnover / max(equity_before, 1.0),
                "invested_ratio": (
                    1.0 - cash / equity_after if equity_after > 0 else 0.0
                ),
                "positions": len(shares),
                "trades": trades,
            }
        )
        previous_equity = equity_after

    daily = pd.DataFrame(rows)
    if record_actions:
        return daily, pd.DataFrame(actions)
    return daily


def local_gate(frame: pd.DataFrame, gate: dict) -> pd.Series:
    return (
        frame["robust_positive"]
        & (frame["full_linear_annual_proxy"] >= gate["linear_annual_proxy_min"])
        & (frame["full_sharpe"] >= gate["sharpe_min"])
        & (frame["full_max_drawdown"] <= gate["max_drawdown_max"])
        & (frame["full_trades"] >= gate["trades_min"])
        & (frame["full_invested_ratio_mean"] >= gate["invested_ratio_min"])
    )


def rank_candidates(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.sort_values(
        [
            "robust_sharpe_floor",
            "min_year_sharpe",
            "full_sharpe",
            "full_linear_annual_proxy",
            "case_id",
        ],
        ascending=[False, False, False, False, True],
    ).head(count)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cache = ROOT / protocol["input_cache"]["path"]
    if protocol["status"] != "frozen_research_only":
        raise RuntimeError("研究协议未冻结")
    if metrics_lib.digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("输入缓存哈希不一致")
    if metrics_lib.digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("研究代码哈希不一致")
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation(
            {key: saved[key] for key in saved.files}, protocol["observation_end"]
        )

    OUT.mkdir(parents=True, exist_ok=True)
    stage1_rows: list[dict] = []
    definitions: dict[str, dict] = {}
    state_cache: dict[str, tuple] = {}
    health_cache: dict[str, np.ndarray] = {}
    fixed_exit = sell_lib.ExitProfile(**protocol["stage1_fixed_exit"])

    for state_profile in protocol["state_profiles"]:
        state_key = stable_id(state_profile)
        base_state = robust.build_state(
            arrays,
            protocol["score_weights"],
            state_profile["agreement"],
            state_profile["entry"]["risk_off"],
            state_profile["entry"]["risk_on"],
            protocol["fixed"],
        )
        quality_state = apply_liquidity(
            base_state,
            arrays,
            PortfolioProfile(
                top_n=5,
                gross_exposure=1.0,
                amount_min=protocol["fixed"]["amount_min"],
                mv_min=protocol["fixed"]["mv_min"],
                turnover_max=protocol["fixed"]["turnover_max"],
            ),
        )
        quality = quality_lib.candidate_quality(
            arrays,
            quality_state,
            int(protocol["health_contract"]["shadow_top_n"]),
            float(protocol["health_contract"]["round_trip_cost"]),
        )
        for values in product(
            protocol["stage1_grid"]["lookback"],
            protocol["stage1_grid"]["mean_threshold"],
            protocol["stage1_grid"]["positive_fraction"],
            protocol["stage1_grid"]["weak_floor"],
        ):
            lookback, mean_threshold, positive_fraction, weak_floor = values
            health_definition = {
                "state_key": state_key,
                "lookback": int(lookback),
                "mean_threshold": float(mean_threshold),
                "positive_fraction": float(positive_fraction),
                "weak_floor": float(weak_floor),
            }
            health_key = stable_id(health_definition)
            health = health_lib.fast_health_exposure(
                quality,
                int(lookback),
                float(mean_threshold),
                float(positive_fraction),
                float(weak_floor),
            )
            health_cache[health_key] = health
            state_cache[health_key] = health_lib.state_with_health(
                quality_state, health
            )
            for top_n, gross_exposure in product(
                protocol["stage1_grid"]["top_n"],
                protocol["stage1_grid"]["gross_exposure"],
            ):
                portfolio = PortfolioProfile(
                    int(top_n),
                    float(gross_exposure),
                    int(protocol["fixed"]["amount_min"]),
                    int(protocol["fixed"]["mv_min"]),
                    float(protocol["fixed"]["turnover_max"]),
                )
                definition = {
                    "state_profile": state_profile,
                    "health": health_definition,
                    "portfolio": asdict(portfolio),
                    "exit": asdict(fixed_exit),
                }
                case_id = "v77d_" + stable_id(definition)
                definitions[case_id] = definition
                daily = simulate(
                    arrays,
                    state_cache[health_key],
                    portfolio,
                    fixed_exit,
                    protocol["observation_end"],
                )
                stage1_rows.append(
                    {
                        "case_id": case_id,
                        "state_id": state_profile["id"],
                        "health_key": health_key,
                        **health_definition,
                        **asdict(portfolio),
                        **asdict(fixed_exit),
                        **robust.evaluate_robust(daily, protocol),
                    }
                )

    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = local_gate(stage1, protocol["stage1_gate"])
    stage1 = stage1.sort_values(
        ["eligible", "robust_sharpe_floor", "full_sharpe", "case_id"],
        ascending=[False, False, False, True],
    )
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = rank_candidates(
        stage1[stage1["eligible"]],
        int(protocol["stage1_grid"]["promote_count"]),
    )

    stage2_rows: list[dict] = []
    stage2_definitions: dict[str, dict] = {}
    for _, seed in promoted.iterrows():
        definition = definitions[str(seed.case_id)]
        portfolio = PortfolioProfile(**definition["portfolio"])
        state = state_cache[str(seed.health_key)]
        for min_hold, max_hold, sell_rank, advantage in product(
            protocol["stage2_grid"]["min_hold"],
            protocol["stage2_grid"]["max_hold"],
            protocol["stage2_grid"]["sell_rank_below"],
            protocol["stage2_grid"]["replacement_advantage"],
        ):
            if int(max_hold) < int(min_hold):
                continue
            exit_profile = sell_lib.ExitProfile(
                "rank_5d",
                int(min_hold),
                int(max_hold),
                float(sell_rank),
                float(advantage),
            )
            stage2_definition = {
                **definition,
                "exit": asdict(exit_profile),
            }
            case_id = "v77e_" + stable_id(stage2_definition)
            stage2_definitions[case_id] = stage2_definition
            daily = simulate(
                arrays,
                state,
                portfolio,
                exit_profile,
                protocol["observation_end"],
            )
            stage2_rows.append(
                {
                    "case_id": case_id,
                    "seed_case_id": str(seed.case_id),
                    "state_id": str(seed.state_id),
                    "health_key": str(seed.health_key),
                    "lookback": int(seed.lookback),
                    "mean_threshold": float(seed.mean_threshold),
                    "positive_fraction": float(seed.positive_fraction),
                    "weak_floor": float(seed.weak_floor),
                    **asdict(portfolio),
                    **asdict(exit_profile),
                    **robust.evaluate_robust(daily, protocol),
                }
            )

    stage2 = pd.DataFrame(stage2_rows)
    if not stage2.empty:
        stage2["eligible"] = local_gate(stage2, protocol["stage2_gate"])
        stage2 = stage2.sort_values(
            ["eligible", "robust_sharpe_floor", "full_sharpe", "case_id"],
            ascending=[False, False, False, True],
        )
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = (
        rank_candidates(
            stage2[stage2["eligible"]],
            int(protocol["stage2_grid"]["freeze_count"]),
        )
        if not stage2.empty
        else stage2
    )

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_rows: list[dict] = []
    seen: set[str] = set()
    for _, item in frozen.iterrows():
        definition = stage2_definitions[str(item.case_id)]
        portfolio = PortfolioProfile(**definition["portfolio"])
        exit_profile = sell_lib.ExitProfile(**definition["exit"])
        _, actions = simulate(
            arrays,
            state_cache[str(item.health_key)],
            portfolio,
            exit_profile,
            protocol["observation_end"],
            record_actions=True,
        )
        content_hash = hashlib.sha256(
            actions.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{item.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_rows.append(
            {
                "case_id": str(item.case_id),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": metrics_lib.digest(path),
                "rows": len(actions),
                "buy_rows": int((actions.action == "BUY").sum()),
                "max_positions": int(portfolio.top_n),
                "per_name_target_pct": portfolio.gross_exposure / portfolio.top_n,
            }
        )

    (OUT / "frozen_candidates.json").write_text(
        json.dumps(
            {
                "protocol_sha256": metrics_lib.digest(PROTOCOL),
                "generated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "profiles": frozen.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps({"actions": action_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": (
            "research_only_observation_candidates_frozen"
            if action_rows
            else "research_only_stopped_at_local_gate"
        ),
        "stage1_cases": len(stage1),
        "stage1_eligible": int(stage1.eligible.sum()),
        "stage1_promoted": len(promoted),
        "stage2_cases": len(stage2),
        "stage2_eligible": (
            int(stage2.eligible.sum()) if not stage2.empty else 0
        ),
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_rows),
        "known_2026_used": False,
        "industry_constraint_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

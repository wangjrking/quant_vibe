# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_best_corrected_v6_20260721 as corrected
import research_active_l4_candidate_strength_v30_20260721 as metrics_lib
import research_active_l4_robust_objective_v56_20260721 as robust
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_10d_fixed_sleeves_v73_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


@dataclass(frozen=True)
class Profile:
    hold_days: int
    max_positions: int
    max_new_positions_per_day: int
    entry_rank_min: float
    gross_exposure: float
    amount_min: int
    mv_min: int
    turnover_max: float

    @property
    def case_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "fs10_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_research_only":
        raise RuntimeError("protocol is not frozen_research_only")
    if digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("code hash mismatch")
    cache = ROOT / protocol["input_cache"]["path"]
    if digest(cache) != protocol["input_cache"]["sha256"]:
        raise RuntimeError("input cache hash mismatch")
    for label, item in protocol["formal_manifests"].items():
        path = ROOT / item["path"]
        if digest(path) != item["sha256"]:
            raise RuntimeError(f"formal manifest hash mismatch: {label}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("approval_status") != "approved_for_l5":
            raise RuntimeError(f"formal manifest not approved_for_l5: {label}")
        if manifest.get("source_type") != "duckdb_table":
            raise RuntimeError(f"formal manifest is not DuckDB: {label}")
    return protocol


def simulate(
    arrays: dict[str, np.ndarray],
    profile: Profile,
    end_date: str,
    record_actions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    dates = arrays["dates"].astype(str)
    stocks = arrays["stocks"].astype(str)
    score = arrays["rank_10d"]
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), axis=1).astype(np.int32)
    board_rate = core.board_limit_rate(stocks)
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

        universe = arrays["signal_clean"][t] & arrays["buy_clean"][t]
        universe &= valid_open & np.isfinite(score[t])
        universe &= score[t] >= profile.entry_rank_min
        universe &= arrays["amount"][t] >= profile.amount_min
        universe &= arrays["total_mv"][t] >= profile.mv_min
        universe &= np.isfinite(arrays["turnover_rate"][t])
        universe &= arrays["turnover_rate"][t] >= 0.0
        universe &= arrays["turnover_rate"][t] <= profile.turnover_max
        universe &= arrays["listed_days"][t] >= 60
        universe &= ~(opens >= pre_close * (1.0 + board_rate) * 0.995)
        ranked = [int(idx) for idx in order[t] if universe[int(idx)]]

        sell_indices = [
            idx for idx in shares if t - entry_index[idx] >= profile.hold_days
        ]
        turnover = 0.0
        trades = 0
        sell_failed = False
        for idx in sell_indices:
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

        available_slots = max(profile.max_positions - len(shares), 0)
        new_slots = min(available_slots, profile.max_new_positions_per_day)
        if sell_failed:
            new_slots = 0
        target_pct = profile.gross_exposure / profile.max_positions
        for idx in ranked:
            if new_slots <= 0:
                break
            if idx in shares:
                continue
            gross_budget = min(equity_before * target_pct, cash / 1.001)
            if gross_budget < 1000:
                break
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
            new_slots -= 1
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


def eligible(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["robust_positive"]
        & (frame["full_max_drawdown"] <= 0.40)
        & (frame["full_trades"] >= 80)
        & (frame["full_invested_ratio_mean"] >= 0.50)
    )


def select_union(frame: pd.DataFrame, robust_count: int, return_count: int) -> pd.DataFrame:
    pool = frame[frame["eligible"]].copy()
    if pool.empty:
        return pool
    robust_top = pool.sort_values(
        ["robust_sharpe_floor", "min_year_sharpe", "full_sharpe", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, False, True],
    ).head(robust_count)
    return_top = pool.sort_values(
        ["full_cumulative_return", "robust_sharpe_floor", "case_id"],
        ascending=[False, False, True],
    ).head(return_count)
    return pd.concat([robust_top, return_top]).drop_duplicates("case_id")


def evaluate(arrays: dict[str, np.ndarray], protocol: dict, profile: Profile) -> dict:
    daily = simulate(arrays, profile, protocol["observation_end"])
    return robust.evaluate_robust(daily, protocol)


def main() -> None:
    protocol = load_protocol()
    cache = ROOT / protocol["input_cache"]["path"]
    with np.load(cache, allow_pickle=False) as saved:
        arrays = robust.truncate_observation(
            {key: saved[key] for key in saved.files}, protocol["observation_end"]
        )
    if np.any(arrays["dates"].astype(str) > protocol["observation_end"]):
        raise RuntimeError("observation truncation failed")
    OUT.mkdir(parents=True, exist_ok=True)

    stage1_rows: list[dict] = []
    stage1_profiles: dict[str, Profile] = {}
    fixed = protocol["stage1_fixed"]
    grid = protocol["stage1_grid"]
    for hold_days in grid["hold_days"]:
        for max_positions in grid["max_positions"]:
            for max_new in grid["max_new_positions_per_day"]:
                for entry_rank_min in grid["entry_rank_min"]:
                    for gross_exposure in grid["gross_exposure"]:
                        profile = Profile(
                            int(hold_days),
                            int(max_positions),
                            int(max_new),
                            float(entry_rank_min),
                            float(gross_exposure),
                            int(fixed["amount_min"]),
                            int(fixed["mv_min"]),
                            float(fixed["turnover_max"]),
                        )
                        stage1_profiles[profile.case_id] = profile
                        stage1_rows.append(
                            {
                                "case_id": profile.case_id,
                                **asdict(profile),
                                **evaluate(arrays, protocol, profile),
                            }
                        )
    stage1 = pd.DataFrame(stage1_rows)
    stage1["eligible"] = eligible(stage1)
    stage1 = stage1.sort_values(
        ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, True],
    )
    stage1.to_csv(OUT / "stage1_grid_results.csv", index=False, encoding="utf-8-sig")
    promoted = select_union(
        stage1,
        int(grid["promote_robust_count"]),
        int(grid["promote_return_count"]),
    )

    stage2_rows: list[dict] = []
    stage2_profiles: dict[str, Profile] = {}
    grid2 = protocol["stage2_grid"]
    for _, seed in promoted.iterrows():
        base = stage1_profiles[str(seed.case_id)]
        for amount_min in grid2["amount_min"]:
            for mv_min in grid2["mv_min"]:
                for turnover_max in grid2["turnover_max"]:
                    profile = Profile(
                        base.hold_days,
                        base.max_positions,
                        base.max_new_positions_per_day,
                        base.entry_rank_min,
                        base.gross_exposure,
                        int(amount_min),
                        int(mv_min),
                        float(turnover_max),
                    )
                    stage2_profiles[profile.case_id] = profile
                    stage2_rows.append(
                        {
                            "case_id": profile.case_id,
                            "seed_case_id": str(seed.case_id),
                            **asdict(profile),
                            **evaluate(arrays, protocol, profile),
                        }
                    )
    stage2 = pd.DataFrame(stage2_rows)
    if not stage2.empty:
        stage2 = stage2.drop_duplicates("case_id")
        stage2["eligible"] = eligible(stage2)
        stage2 = stage2.sort_values(
            ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
            ascending=[False, False, False, True],
        )
    stage2.to_csv(OUT / "stage2_grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = (
        select_union(
            stage2,
            int(grid2["freeze_robust_count"]),
            int(grid2["freeze_return_count"]),
        )
        if not stage2.empty
        else stage2
    )

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_manifest: list[dict] = []
    seen: set[str] = set()
    for _, row in frozen.iterrows():
        profile = stage2_profiles[str(row.case_id)]
        _, actions = simulate(
            arrays, profile, protocol["observation_end"], record_actions=True
        )
        content_hash = hashlib.sha256(
            actions.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{profile.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_manifest.append(
            {
                "case_id": profile.case_id,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": digest(path),
                "rows": len(actions),
                "buy_rows": int((actions.action == "BUY").sum()),
                "max_positions": profile.max_positions,
            }
        )

    (OUT / "frozen_juejin_candidates.json").write_text(
        json.dumps(
            {
                "protocol_sha256": digest(PROTOCOL),
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "profiles": frozen.to_dict("records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "juejin_action_manifest.json").write_text(
        json.dumps({"actions": action_manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "status": (
            "research_only_observation_candidates_frozen"
            if action_manifest
            else "research_only_stopped_at_local_gate"
        ),
        "stage1_cases": len(stage1),
        "stage1_eligible": int(stage1.eligible.sum()),
        "stage1_promoted": len(promoted),
        "stage2_cases": len(stage2),
        "stage2_eligible": int(stage2.eligible.sum()) if not stage2.empty else 0,
        "frozen_candidates": len(frozen),
        "unique_juejin_paths": len(action_manifest),
        "known_2026_used": False,
        "production_changed": False,
    }
    (OUT / "research_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

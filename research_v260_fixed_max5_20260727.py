from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import export_v260_all4key_production_signals as production_exporter
import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = (
    ROOT
    / "quant/main/strategy_library/production/"
    "prod_v260_10d_regime_warmup_all4key_v20260724"
)
OUT = (
    ROOT
    / "quant/data_file/reports/"
    "strategy_agent_v260_fixed_max5_20260727"
)
PROTOCOL = STRATEGY_DIR / "inputs/preregistered_protocol.json"
BACKTEST_SIGNAL_END = "20260721"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def max_active_positions(actions: pd.DataFrame) -> int:
    active: set[str] = set()
    maximum = 0
    ordered = actions.copy()
    ordered["_action_order"] = ordered["action"].map({"SELL": 0, "BUY": 1})
    ordered = ordered.sort_values(
        ["signal_date", "_action_order", "stock_code"],
        kind="stable",
    )
    for row in ordered.itertuples(index=False):
        if row.action == "SELL":
            active.discard(str(row.stock_code))
        elif row.action == "BUY":
            active.add(str(row.stock_code))
        maximum = max(maximum, len(active))
    return maximum


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = production_exporter.validate_route()
    arrays = production_exporter.build_arrays(manifests)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    definition = v260.definition_for(protocol, 50)
    definition.update(
        {
            "max_positions": 5,
            "weak_max_positions": 5,
            "normal_max_positions": 5,
            "strong_max_positions": 5,
            "position_warmup_days": 0,
        }
    )

    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    fixed_schedule = np.full(len(arrays["dates"]), 5, dtype=np.int16)
    daily, actions = v260.v258.v252.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        BACKTEST_SIGNAL_END,
        record_actions=True,
        max_positions_schedule_override=fixed_schedule,
    )

    action_path = OUT / "v260_fixed_max5_actions.csv"
    daily_path = OUT / "v260_fixed_max5_daily_local.csv"
    actions.to_csv(action_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    duplicate_actions = int(
        actions.duplicated(
            ["signal_date", "buy_date", "action", "stock_code"]
        ).sum()
    )
    bj_rows = int(actions["stock_code"].astype(str).str.endswith(".BJ").sum())
    observed_max_positions = max_active_positions(actions)
    if duplicate_actions:
        raise RuntimeError(f"duplicate action keys: {duplicate_actions}")
    if bj_rows:
        raise RuntimeError(f"BJ action rows: {bj_rows}")
    if observed_max_positions > 5:
        raise RuntimeError(
            f"max-position contract failed: observed {observed_max_positions}"
        )

    local_metrics = core.metrics(
        daily,
        str(arrays["dates"][0]),
        BACKTEST_SIGNAL_END,
    )
    report = {
        "schema_version": 1,
        "status": "research_only_not_production",
        "base_strategy_id": (
            "prod_v260_10d_regime_warmup_all4key_v20260724"
        ),
        "candidate_id": "research_v260_fixed_max5_20260727",
        "single_change": {
            "field": "maximum_concurrent_positions",
            "production_value": "startup 16; weak/normal/strong 14/15/16",
            "research_value": 5,
        },
        "unchanged_contract": {
            "key_domain": "inner_intersection_1d_3d_5d_10d",
            "score": "10D pred_prob rank only",
            "smooth_window_trade_days": 7,
            "current_weight": 0.1,
            "fixed_slippage_each_side": 0.003,
            "commission_each_side": 0.0003,
            "stamp_duty_sell": 0.0005,
            "initial_cash": 700000,
        },
        "formal_max_trade_date": str(arrays["dates"][-1]),
        "backtest_signal_end": BACKTEST_SIGNAL_END,
        "action_rows": int(len(actions)),
        "buy_rows": int((actions["action"] == "BUY").sum()),
        "sell_rows": int((actions["action"] == "SELL").sum()),
        "duplicate_action_keys": duplicate_actions,
        "bj_rows": bj_rows,
        "observed_max_active_positions": observed_max_positions,
        "action_sha256": sha256(action_path),
        "daily_sha256": sha256(daily_path),
        "local_screening_metrics": local_metrics,
        "formal_manifests": {
            label: str(path)
            for label, path in production_exporter.FORMAL_MANIFESTS.items()
        },
        "outputs": {
            "actions": str(action_path),
            "daily": str(daily_path),
        },
        "juejin_backtest": {
            "status": "pending",
            "backtest_start": "2022-06-07 09:00:00",
            "backtest_end": "2026-07-22 15:30:00",
        },
        "production_modified": False,
        "l5_l8_modified": False,
    }
    report_path = OUT / "research_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

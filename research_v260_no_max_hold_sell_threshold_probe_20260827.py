from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_v260_hold_period_sell_threshold_probe_20260827 as prior


ROOT = Path(__file__).resolve().parents[2]
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_v260_no_max_hold_sell_threshold_probe_20260827"
)
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
NO_MAX_HOLD_DAYS = 10000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(prior.PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache_path = ROOT / protocol["input_cache"]
    with np.load(cache_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    base_definition = v260.definition_for(protocol, 50)

    baseline_development, _, _ = prior.run_window(
        arrays,
        score,
        order,
        protocol,
        base_definition,
        prior.DEV_START,
        prior.DEV_END,
    )
    baseline_validation, _, _ = prior.run_window(
        arrays,
        score,
        order,
        protocol,
        base_definition,
        prior.VALIDATION_START,
        prior.VALIDATION_END,
    )

    cases = []
    for threshold in THRESHOLDS:
        definition = {
            **base_definition,
            "max_hold_days": NO_MAX_HOLD_DAYS,
            "sell_score_below": float(threshold),
        }
        development, _, _ = prior.run_window(
            arrays,
            score,
            order,
            protocol,
            definition,
            prior.DEV_START,
            prior.DEV_END,
        )
        validation, _, _ = prior.run_window(
            arrays,
            score,
            order,
            protocol,
            definition,
            prior.VALIDATION_START,
            prior.VALIDATION_END,
        )
        cases.append(
            {
                "case_id": f"no_max_hold_sell_{threshold:.2f}".replace(".", "p"),
                "sell_score_below": threshold,
                "definition": definition,
                "development": development,
                "validation_2026": validation,
                "development_vs_production": {
                    key: float(development[key] - baseline_development[key])
                    for key in (
                        "cumulative_return",
                        "cagr",
                        "sharpe",
                        "max_drawdown",
                        "turnover",
                    )
                },
                "validation_vs_production": {
                    key: float(validation[key] - baseline_validation[key])
                    for key in (
                        "cumulative_return",
                        "cagr",
                        "sharpe",
                        "max_drawdown",
                        "turnover",
                    )
                },
            }
        )

    development_selected = sorted(
        cases,
        key=lambda item: (
            item["development"]["sharpe"],
            item["development"]["cumulative_return"],
        ),
        reverse=True,
    )[0]
    validation_leader = sorted(
        cases,
        key=lambda item: (
            item["validation_2026"]["sharpe"],
            item["validation_2026"]["cumulative_return"],
        ),
        reverse=True,
    )[0]

    replay, _, _ = prior.run_window(
        arrays,
        score,
        order,
        protocol,
        development_selected["definition"],
        prior.VALIDATION_START,
        prior.VALIDATION_END,
    )
    deterministic = all(
        replay[key] == development_selected["validation_2026"][key]
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover",
            "trades",
        )
    )

    rows = []
    for item in cases:
        rows.append(
            {
                "case_id": item["case_id"],
                "sell_score_below": item["sell_score_below"],
                **{
                    f"development_{key}": item["development"][key]
                    for key in (
                        "cumulative_return",
                        "cagr",
                        "sharpe",
                        "max_drawdown",
                        "turnover",
                        "invested_ratio_mean",
                        "average_hold_trade_days",
                    )
                },
                **{
                    f"validation_2026_{key}": item["validation_2026"][key]
                    for key in (
                        "cumulative_return",
                        "cagr",
                        "sharpe",
                        "max_drawdown",
                        "turnover",
                        "invested_ratio_mean",
                        "average_hold_trade_days",
                    )
                },
            }
        )
    pd.DataFrame(rows).sort_values(
        ["development_sharpe", "development_cumulative_return"],
        ascending=[False, False],
    ).to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": 1,
        "task": "remove_max_hold_and_select_sell_score_threshold",
        "selection_policy": "select by development Sharpe, then development cumulative return; 2026 is validation only",
        "development_window": [prior.DEV_START, prior.DEV_END],
        "validation_window": [prior.VALIDATION_START, prior.VALIDATION_END],
        "max_hold_days": NO_MAX_HOLD_DAYS,
        "max_hold_semantics": "effectively disabled",
        "thresholds": list(THRESHOLDS),
        "production_baseline": {
            "max_hold_days": 20,
            "sell_score_below": 0.85,
            "development": baseline_development,
            "validation_2026": baseline_validation,
        },
        "development_selected_case_id": development_selected["case_id"],
        "validation_leader_case_id_for_diagnostics_only": validation_leader["case_id"],
        "selected_case_deterministic_replay": deterministic,
        "cases": cases,
        "production_modified": False,
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "development_selected": development_selected["case_id"],
                "validation_leader_diagnostic": validation_leader["case_id"],
                "deterministic": deterministic,
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

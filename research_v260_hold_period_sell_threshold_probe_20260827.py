from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "quant/data_file/reports/strategy_agent_active_l4_v258_position_warmup_v260_20260723"
)
PROTOCOL_PATH = SOURCE / "preregistered_protocol.json"
OUT = (
    ROOT
    / "quant/data_file/reports/strategy_agent_v260_hold_period_sell_threshold_probe_20260827"
)

DEV_START = "20220607"
DEV_END = "20251231"
VALIDATION_START = "20260105"
VALIDATION_END = "20260720"
MAX_HOLD_DAYS = (20, 30, 40, 60)
SELL_THRESHOLDS = (0.85, 0.90)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def annual_returns(daily: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    frame = daily[
        (daily["date"].astype(str) >= str(start))
        & (daily["date"].astype(str) <= str(end))
    ].copy()
    frame["year"] = frame["date"].astype(str).str[:4]
    return {
        str(year): float(np.prod(1.0 + group["return"].astype(float)) - 1.0)
        for year, group in frame.groupby("year", sort=True)
    }


def holding_stats(actions: pd.DataFrame, dates: np.ndarray) -> dict[str, float | int | None]:
    date_index = {str(date): index for index, date in enumerate(dates.astype(str))}
    entries: dict[str, int] = {}
    ages: list[int] = []
    for row in actions.itertuples(index=False):
        stock = str(row.stock_code)
        signal_date = str(row.signal_date)
        if row.action == "BUY":
            entries[stock] = date_index[signal_date]
        elif row.action == "SELL" and stock in entries:
            ages.append(date_index[signal_date] - entries.pop(stock))
    return {
        "closed_round_trips": len(ages),
        "average_hold_trade_days": float(np.mean(ages)) if ages else None,
        "median_hold_trade_days": float(np.median(ages)) if ages else None,
        "p90_hold_trade_days": float(np.percentile(ages, 90)) if ages else None,
    }


def run_window(
    arrays: dict[str, np.ndarray],
    score: np.ndarray,
    order: np.ndarray,
    protocol: dict,
    definition: dict,
    start: str,
    end: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    daily, actions = v260.run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        end,
        start=start,
        record_actions=True,
    )
    result = {
        **core.metrics(daily, start, end),
        **holding_stats(actions, arrays["dates"]),
        "annual_returns": annual_returns(daily, start, end),
        "buy_actions": int((actions["action"] == "BUY").sum()),
        "sell_actions": int((actions["action"] == "SELL").sum()),
    }
    return result, daily, actions


def case_id(max_hold_days: int, sell_score_below: float) -> str:
    return f"hold{max_hold_days}_sell{sell_score_below:.2f}".replace(".", "p")


def flatten(case: dict) -> dict:
    row = {
        "case_id": case["case_id"],
        "max_hold_days": case["max_hold_days"],
        "sell_score_below": case["sell_score_below"],
    }
    for window in ("development", "validation_2026"):
        metrics = case[window]
        for key in (
            "cumulative_return",
            "cagr",
            "linear_annual_proxy",
            "sharpe",
            "max_drawdown",
            "turnover",
            "invested_ratio_mean",
            "trades",
            "closed_round_trips",
            "average_hold_trade_days",
            "median_hold_trade_days",
            "p90_hold_trade_days",
        ):
            row[f"{window}_{key}"] = metrics.get(key)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    cache_path = ROOT / protocol["input_cache"]
    with np.load(cache_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    score, order = v95.score_pair(arrays, 0.0, 7, 0.1)
    baseline_definition = v260.definition_for(protocol, 50)

    cases = []
    for max_hold_days in MAX_HOLD_DAYS:
        for threshold in SELL_THRESHOLDS:
            definition = {
                **baseline_definition,
                "max_hold_days": int(max_hold_days),
                "sell_score_below": float(threshold),
            }
            development, _, _ = run_window(
                arrays,
                score,
                order,
                protocol,
                definition,
                DEV_START,
                DEV_END,
            )
            validation, _, _ = run_window(
                arrays,
                score,
                order,
                protocol,
                definition,
                VALIDATION_START,
                VALIDATION_END,
            )
            cases.append(
                {
                    "case_id": case_id(max_hold_days, threshold),
                    "max_hold_days": max_hold_days,
                    "sell_score_below": threshold,
                    "definition": definition,
                    "development": development,
                    "validation_2026": validation,
                }
            )

    baseline = next(
        item
        for item in cases
        if item["max_hold_days"] == 20 and item["sell_score_below"] == 0.85
    )
    for item in cases:
        for window in ("development", "validation_2026"):
            item[f"{window}_delta"] = {
                key: float(item[window][key] - baseline[window][key])
                for key in (
                    "cumulative_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "turnover",
                    "invested_ratio_mean",
                )
            }

    frame = pd.DataFrame(flatten(item) for item in cases)
    frame = frame.sort_values(
        ["validation_2026_cumulative_return", "validation_2026_sharpe"],
        ascending=[False, False],
    )
    frame.to_csv(OUT / "comparison.csv", index=False, encoding="utf-8-sig")

    top_case_id = str(frame.iloc[0]["case_id"])
    top_case = next(item for item in cases if item["case_id"] == top_case_id)
    replay, _, _ = run_window(
        arrays,
        score,
        order,
        protocol,
        top_case["definition"],
        VALIDATION_START,
        VALIDATION_END,
    )
    determinism = {
        key: replay[key] == top_case["validation_2026"][key]
        for key in (
            "cumulative_return",
            "cagr",
            "sharpe",
            "max_drawdown",
            "turnover",
            "trades",
        )
    }

    payload = {
        "schema_version": 1,
        "task": "v260_max_hold_and_sell_score_threshold_probe",
        "production_modified": False,
        "development_window": [DEV_START, DEV_END],
        "validation_window": [VALIDATION_START, VALIDATION_END],
        "score_semantics": {
            "0.85": "sell after minimum hold when score falls outside top 15 percent",
            "0.90": "sell after minimum hold when score falls outside top 10 percent",
        },
        "input_cache": str(cache_path),
        "input_cache_sha256": sha256(cache_path),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "baseline_case_id": baseline["case_id"],
        "validation_leader_case_id": top_case_id,
        "validation_leader_determinism": determinism,
        "cases": cases,
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "baseline": baseline["case_id"],
                "validation_leader": top_case_id,
                "deterministic": all(determinism.values()),
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

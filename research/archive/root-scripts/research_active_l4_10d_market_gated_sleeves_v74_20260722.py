# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import research_active_l4_10d_fixed_sleeves_v73_20260722 as sleeves
import research_active_l4_continuous_market_v53_20260721 as continuous
import research_active_l4_market_state_v26_20260721 as market
import research_active_l4_robust_objective_v56_20260721 as robust


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "quant/data_file/reports/strategy_agent_active_l4_10d_market_gated_sleeves_v74_20260722"
PROTOCOL = OUT / "preregistered_protocol.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stable_id(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "mg10_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_research_only":
        raise RuntimeError("protocol is not frozen_research_only")
    if digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("code hash mismatch")
    dependency = ROOT / protocol["dependency"]["path"]
    if digest(dependency) != protocol["dependency"]["sha256"]:
        raise RuntimeError("fixed-sleeve dependency hash mismatch")
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


def apply_market_gate(
    arrays: dict[str, np.ndarray], lookback: int, threshold: float
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    market_return = market.market_open_return(arrays)
    market_mean = continuous.rolling_mean(market_return, lookback)
    gate = np.isfinite(market_mean) & (market_mean > threshold)
    gated = dict(arrays)
    gated["signal_clean"] = arrays["signal_clean"] & gate[:, None]
    return gated, gate


def eligible(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["robust_positive"]
        & (frame["full_max_drawdown"] <= 0.40)
        & (frame["full_trades"] >= 80)
        & (frame["full_invested_ratio_mean"] >= 0.20)
        & (frame["market_on_fraction"] >= 0.20)
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

    rows: list[dict] = []
    definitions: dict[str, dict] = {}
    gated_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    grid = protocol["grid"]
    for lookback in grid["market_lookback"]:
        for threshold in grid["market_mean_threshold"]:
            gate_key = f"lb{lookback}_th{threshold}"
            gated_cache[gate_key] = apply_market_gate(
                arrays, int(lookback), float(threshold)
            )
            gated, gate = gated_cache[gate_key]
            for hold_days in grid["hold_days"]:
                for max_positions in grid["max_positions"]:
                    for max_new in grid["max_new_positions_per_day"]:
                        for entry_rank_min in grid["entry_rank_min"]:
                            for gross_exposure in grid["gross_exposure"]:
                                profile = sleeves.Profile(
                                    int(hold_days), int(max_positions), int(max_new),
                                    float(entry_rank_min), float(gross_exposure),
                                    int(protocol["fixed_liquidity"]["amount_min"]),
                                    int(protocol["fixed_liquidity"]["mv_min"]),
                                    float(protocol["fixed_liquidity"]["turnover_max"]),
                                )
                                definition = {
                                    "gate_key": gate_key,
                                    "market_lookback": int(lookback),
                                    "market_mean_threshold": float(threshold),
                                    "profile": asdict(profile),
                                }
                                case_id = stable_id(definition)
                                definitions[case_id] = definition
                                daily = sleeves.simulate(
                                    gated, profile, protocol["observation_end"]
                                )
                                values = robust.evaluate_robust(daily, protocol)
                                rows.append(
                                    {
                                        "case_id": case_id,
                                        "gate_key": gate_key,
                                        "market_lookback": int(lookback),
                                        "market_mean_threshold": float(threshold),
                                        "market_on_fraction": float(np.mean(gate)),
                                        **asdict(profile),
                                        **values,
                                    }
                                )

    results = pd.DataFrame(rows)
    results["eligible"] = eligible(results)
    results = results.sort_values(
        ["eligible", "robust_sharpe_floor", "full_cumulative_return", "case_id"],
        ascending=[False, False, False, True],
    )
    results.to_csv(OUT / "grid_results.csv", index=False, encoding="utf-8-sig")
    frozen = select_union(
        results,
        int(protocol["selection"]["freeze_robust_count"]),
        int(protocol["selection"]["freeze_return_count"]),
    )

    action_dir = OUT / "juejin_actions"
    action_dir.mkdir(parents=True, exist_ok=True)
    action_manifest: list[dict] = []
    seen: set[str] = set()
    for _, row in frozen.iterrows():
        definition = definitions[str(row.case_id)]
        profile = sleeves.Profile(**definition["profile"])
        gated = gated_cache[definition["gate_key"]][0]
        _, actions = sleeves.simulate(
            gated, profile, protocol["observation_end"], record_actions=True
        )
        content_hash = hashlib.sha256(
            actions.to_csv(index=False).encode("utf-8")
        ).hexdigest()
        if content_hash in seen:
            continue
        seen.add(content_hash)
        path = action_dir / f"{row.case_id}.csv"
        actions.to_csv(path, index=False, encoding="utf-8-sig")
        action_manifest.append(
            {
                "case_id": str(row.case_id),
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
        "grid_cases": len(results),
        "eligible_cases": int(results.eligible.sum()),
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

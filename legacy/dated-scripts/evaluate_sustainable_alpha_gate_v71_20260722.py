from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "quant/main/config/strategy_research_sustainable_alpha_sleeves_v71_20260722.json"
CACHE = ROOT / "quant/data_file/reports/strategy_agent_active_l4_liquidity_risk_v59_20260721/active_formal_rank_liquidity_arrays_v2.npz"
OUT = ROOT / "quant/data_file/reports/strategy_agent_sustainable_alpha_sleeves_v71_20260722"
CACHE_SHA256 = "2ca8427ef302ba35ffb72652458fbccf7b60a29b81fda9b0a97b24c8c70ef704"
HORIZONS = {"1d": 1, "3d": 3, "5d": 5, "10d": 10}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def period_mask(dates: np.ndarray, name: str) -> np.ndarray:
    if name == "2023":
        return (dates >= "20230101") & (dates <= "20231231")
    if name == "2024":
        return (dates >= "20240101") & (dates <= "20241231")
    if name == "2025":
        return (dates >= "20250101") & (dates <= "20251231")
    raise ValueError(name)


def stable_top(scores: np.ndarray, valid: np.ndarray, topn: int) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return indices
    order = np.argsort(-scores[indices], kind="stable")
    return indices[order[:topn]]


def summarize_period(frame: pd.DataFrame, dates: np.ndarray, name: str) -> dict:
    if name == "recent120":
        available = sorted(frame.loc[frame["trade_date"] <= "20251231", "trade_date"].unique())
        selected = set(available[-120:])
        part = frame[frame["trade_date"].isin(selected)]
    else:
        selected = set(dates[period_mask(dates, name)])
        part = frame[frame["trade_date"].isin(selected)]
    return {
        "days": int(part["trade_date"].nunique()),
        "gross_top3_mean": float(part["gross_top3"].mean()) if len(part) else None,
        "net_top3_mean": float(part["net_top3"].mean()) if len(part) else None,
        "positive_day_fraction": float((part["net_top3"] > 0).mean()) if len(part) else None,
    }


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_waiting_for_independent_alpha_gate":
        raise RuntimeError("V71研究协议状态不允许执行Alpha门")
    if digest(CACHE) != CACHE_SHA256:
        raise RuntimeError("正式L4派生缓存哈希不一致")

    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(CACHE, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}

    dates = arrays["dates"].astype(str)
    observation = dates <= protocol["time_contract"]["strategy_selection_end"]
    base_valid = (
        arrays["signal_clean"]
        & arrays["buy_clean"]
        & np.isfinite(arrays["buy_open"])
        & (arrays["buy_open"] > 0)
        & np.isfinite(arrays["amount"])
        & (arrays["amount"] >= 90000.0)
        & np.isfinite(arrays["total_mv"])
        & (arrays["total_mv"] >= 200000.0)
        & (arrays["listed_days"] >= 60)
    )
    cost = float(protocol["alpha_gate"]["round_trip_cost"])
    topn = int(protocol["alpha_gate"]["topn"])
    daily_rows: list[dict] = []
    picks: dict[str, dict[int, set[int]]] = {key: {} for key in HORIZONS}

    for label, holding_days in HORIZONS.items():
        scores = arrays[f"rank_{label}"]
        for day in range(len(dates) - holding_days):
            if not observation[day]:
                continue
            future_open = arrays["buy_open"][day + holding_days]
            valid = (
                base_valid[day]
                & np.isfinite(scores[day])
                & np.isfinite(future_open)
                & (future_open > 0)
            )
            selected = stable_top(scores[day], valid, topn)
            if len(selected) < topn:
                continue
            gross = float(np.mean(future_open[selected] / arrays["buy_open"][day, selected] - 1.0))
            daily_rows.append(
                {
                    "horizon": label,
                    "trade_date": dates[day],
                    "gross_top3": gross,
                    "net_top3": gross - cost,
                }
            )
            picks[label][day] = set(int(value) for value in selected)

    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / "alpha_gate_daily_top3.csv", index=False, encoding="utf-8-sig")

    horizon_rows = []
    required_periods = list(protocol["alpha_gate"]["required_positive_periods"])
    for label in HORIZONS:
        frame = daily[daily["horizon"] == label]
        periods = {name: summarize_period(frame, dates, name) for name in required_periods}
        passed = all(
            periods[name]["net_top3_mean"] is not None
            and periods[name]["net_top3_mean"] > 0
            for name in required_periods
        )
        horizon_rows.append({"horizon": label, "periods": periods, "positive_period_gate": passed})

    pair_rows = []
    for left, right in combinations(HORIZONS, 2):
        common_days = sorted(set(picks[left]) & set(picks[right]))
        overlaps = []
        correlations = []
        left_score = arrays[f"rank_{left}"]
        right_score = arrays[f"rank_{right}"]
        for day in common_days:
            overlaps.append(len(picks[left][day] & picks[right][day]) / topn)
            valid = base_valid[day] & np.isfinite(left_score[day]) & np.isfinite(right_score[day])
            if int(valid.sum()) >= 30:
                correlations.append(float(np.corrcoef(left_score[day, valid], right_score[day, valid])[0, 1]))
        pair_rows.append(
            {
                "left": left,
                "right": right,
                "common_days": len(common_days),
                "mean_top3_overlap": float(np.mean(overlaps)) if overlaps else None,
                "mean_daily_rank_correlation": float(np.nanmean(correlations)) if correlations else None,
            }
        )

    passing_horizons = {row["horizon"] for row in horizon_rows if row["positive_period_gate"]}
    passing_pairs = []
    for row in pair_rows:
        if row["left"] not in passing_horizons or row["right"] not in passing_horizons:
            continue
        if row["mean_top3_overlap"] is None or row["mean_daily_rank_correlation"] is None:
            continue
        if (
            row["mean_top3_overlap"] <= float(protocol["alpha_gate"]["max_pairwise_top3_overlap"])
            and row["mean_daily_rank_correlation"] <= float(protocol["alpha_gate"]["max_pairwise_daily_rank_correlation"])
        ):
            passing_pairs.append([row["left"], row["right"]])

    gate_pass = (
        len(passing_horizons) >= int(protocol["alpha_gate"]["required_min_sources"])
        and bool(passing_pairs)
    )
    result = {
        "protocol_id": protocol["protocol_id"],
        "status": "alpha_gate_passed" if gate_pass else "alpha_gate_failed",
        "input_cache": str(CACHE.relative_to(ROOT)).replace("\\", "/"),
        "input_cache_sha256": CACHE_SHA256,
        "input_lineage": "由当前active formal L4四份manifest和正式L2 DuckDB按冻结协议生成，不含旧信号名单",
        "observation_end": protocol["time_contract"]["strategy_selection_end"],
        "known_2026_used": False,
        "round_trip_cost": cost,
        "horizons": horizon_rows,
        "pairs": pair_rows,
        "passing_horizons": sorted(passing_horizons),
        "passing_pairs": passing_pairs,
        "strategy_grid_allowed": gate_pass,
        "production_changed": False,
        "code_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(PROTOCOL),
    }
    (OUT / "alpha_gate_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(pair_rows).to_csv(OUT / "alpha_gate_pair_summary.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# V71 独立 Alpha 门结果",
        "",
        f"- 结论：`{result['status']}`",
        f"- 通过期限：{', '.join(result['passing_horizons']) or '无'}",
        f"- 通过组合：{json.dumps(result['passing_pairs'], ensure_ascii=False)}",
        "- 2026参与选参：否",
        "- 正式资产修改：否",
        "",
        "| 期限 | 2023净Top3 | 2024净Top3 | 2025净Top3 | recent120净Top3 | 通过 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in horizon_rows:
        p = row["periods"]
        lines.append(
            f"| {row['horizon']} | {p['2023']['net_top3_mean']:.4%} | {p['2024']['net_top3_mean']:.4%} | "
            f"{p['2025']['net_top3_mean']:.4%} | {p['recent120']['net_top3_mean']:.4%} | "
            f"{'是' if row['positive_period_gate'] else '否'} |"
        )
    lines.extend([
        "",
        "该结果是策略网格前的成本后Alpha粗门，不是掘金策略回测收益。只有本门通过，才允许按冻结V71协议生成候选并送掘金。",
    ])
    (OUT / "独立Alpha门结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passing_horizons": result["passing_horizons"], "passing_pairs": passing_pairs}, ensure_ascii=False))


if __name__ == "__main__":
    main()

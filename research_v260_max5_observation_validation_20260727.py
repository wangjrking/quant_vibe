from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import export_v260_all4key_production_signals as production_exporter
import research_active_l4_10d_smoothing_refine_v95_20260722 as v95
import research_active_l4_market_regime_v134_20260722 as v134
import research_active_l4_v258_position_warmup_v260_20260723 as v260
import research_preregistered_active_l4_rank_rotation_20260721 as core


ROOT = Path(__file__).resolve().parents[2]
ROUND = os.environ.get("V260_MAX5_RESEARCH_ROUND", "round1").strip()
OUT_SUFFIX = "" if ROUND == "round1" else f"_{ROUND}"
OUT = ROOT / (
    "quant/data_file/reports/"
    f"strategy_agent_v260_max5_observation_validation_20260727{OUT_SUFFIX}"
)
PROTOCOL = OUT / "预注册协议.json"
FROZEN = OUT / "观察期冻结候选.json"


def stable_id(definition: dict) -> str:
    raw = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return "max5_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _legacy_load_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    manifests = production_exporter.validate_route()
    arrays = production_exporter.build_arrays(manifests)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    score, order = v95.score_pair(
        arrays,
        0.0,
        int(protocol["固定参数"]["平滑交易日"]),
        float(protocol["固定参数"]["当日权重"]),
    )
    ensemble_weight = protocol["固定参数"].get("smooth_7d_ensemble_weight")
    if ensemble_weight is not None:
        score_6d, _ = v95.score_pair(
            arrays,
            0.0,
            6,
            float(protocol["固定参数"]["当日权重"]),
        )
        weight_7d = float(ensemble_weight)
        score = (
            weight_7d * score + (1.0 - weight_7d) * score_6d
        ).astype(np.float32)
        order = np.argsort(
            -np.nan_to_num(score, nan=-np.inf),
            axis=1,
            kind="stable",
        )
    secondary_window = protocol["鍥哄畾鍙傛暟"].get(
        "smooth_secondary_window"
    )
    if secondary_window is not None:
        secondary_score, _ = v95.score_pair(
            arrays,
            0.0,
            int(secondary_window),
            float(protocol["鍥哄畾鍙傛暟"]["褰撴棩鏉冮噸"]),
        )
        primary_weight = float(
            protocol["鍥哄畾鍙傛暟"]["smooth_primary_weight"]
        )
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("smooth_primary_weight must be in [0, 1]")
        score = (
            primary_weight * score
            + (1.0 - primary_weight) * secondary_score
        ).astype(np.float32)
        order = np.argsort(
            -np.nan_to_num(score, nan=-np.inf),
            axis=1,
            kind="stable",
        )
    return arrays, protocol, score, order


def load_inputs() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    manifests = production_exporter.validate_route()
    arrays = production_exporter.build_arrays(manifests)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    fixed = protocol["固定参数"]
    score_window = int(fixed["平滑交易日"])
    current_weight = float(fixed["当日权重"])
    score, order = v95.score_pair(
        arrays,
        0.0,
        score_window,
        current_weight,
    )
    ensemble_weight = fixed.get("smooth_7d_ensemble_weight")
    if ensemble_weight is not None:
        score_6d, _ = v95.score_pair(
            arrays,
            0.0,
            6,
            current_weight,
        )
        weight_7d = float(ensemble_weight)
        score = (
            weight_7d * score + (1.0 - weight_7d) * score_6d
        ).astype(np.float32)
        order = np.argsort(
            -np.nan_to_num(score, nan=-np.inf),
            axis=1,
            kind="stable",
        )
    secondary_window = fixed.get("smooth_secondary_window")
    if secondary_window is not None:
        secondary_score, _ = v95.score_pair(
            arrays,
            0.0,
            int(secondary_window),
            current_weight,
        )
        primary_weight = float(fixed["smooth_primary_weight"])
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("smooth_primary_weight must be in [0, 1]")
        score = (
            primary_weight * score
            + (1.0 - primary_weight) * secondary_score
        ).astype(np.float32)
        order = np.argsort(
            -np.nan_to_num(score, nan=-np.inf),
            axis=1,
            kind="stable",
        )
    return arrays, protocol, score, order


def definition_for(protocol: dict, values: tuple) -> dict:
    (
        normal_target,
        weak_target,
        strong_target,
        sell_below,
        advantage,
        min_hold_policy,
        max_hold_days,
    ) = values
    base_protocol = json.loads(
        (
            ROOT
            / "quant/main/strategy_library/production/"
            "prod_v260_10d_regime_warmup_all4key_v20260724/"
            "inputs/preregistered_protocol.json"
        ).read_text(encoding="utf-8")
    )
    definition = v260.definition_for(base_protocol, 0)
    definition.update(
        {
            "max_positions": 5,
            "weak_max_positions": 5,
            "normal_max_positions": 5,
            "strong_max_positions": 5,
            "position_warmup_days": 0,
            "smooth_window": int(protocol["固定参数"]["平滑交易日"]),
            "current_weight": float(protocol["固定参数"]["当日权重"]),
            "normal_target_pct": float(normal_target),
            "weak_target_pct": float(weak_target),
            "strong_target_pct": float(strong_target),
            "sell_score_below": float(sell_below),
            "replacement_advantage": float(advantage),
            "min_hold_policy": str(min_hold_policy),
            "max_hold_days": int(max_hold_days),
        }
    )
    fixed = protocol["固定参数"]
    if "强势收益阈值" in fixed:
        definition["strong_return_min"] = float(fixed["强势收益阈值"])
    if "市场观察交易日" in fixed:
        definition["market_lookback"] = int(fixed["市场观察交易日"])
    if "市场收益分界" in fixed:
        definition["market_return_min"] = float(fixed["市场收益分界"])
    if "strong_return_min" in fixed:
        definition["strong_return_min"] = float(fixed["strong_return_min"])
    if "market_lookback" in fixed:
        definition["market_lookback"] = int(fixed["market_lookback"])
    if "market_return_min" in fixed:
        definition["market_return_min"] = float(fixed["market_return_min"])
    if "sell_score_mode" in fixed:
        definition["sell_score_mode"] = str(fixed["sell_score_mode"])
    if "sell_score_regime_offsets" in fixed:
        definition["sell_score_regime_offsets"] = [
            float(value) for value in fixed["sell_score_regime_offsets"]
        ]
    if "sell_score_quantile_blend" in fixed:
        definition["sell_score_quantile_blend"] = [
            float(value) for value in fixed["sell_score_quantile_blend"]
        ]
    if "sell_score_quantile_adaptive" in fixed:
        definition["sell_score_quantile_adaptive"] = [
            float(value) for value in fixed["sell_score_quantile_adaptive"]
        ]
    if "replacement_advantage_dispersion_adaptive" in fixed:
        definition["replacement_advantage_dispersion_adaptive"] = [
            float(value)
            for value in fixed["replacement_advantage_dispersion_adaptive"]
        ]
    if "max_rank_deterioration" in fixed:
        definition["max_rank_deterioration"] = float(
            fixed["max_rank_deterioration"]
        )
    if "low_gross" in fixed:
        definition["low_gross"] = float(fixed["low_gross"])
    if "smooth_7d_ensemble_weight" in fixed:
        definition["smooth_7d_ensemble_weight"] = float(
            fixed["smooth_7d_ensemble_weight"]
        )
    if "smooth_secondary_window" in fixed:
        definition["smooth_secondary_window"] = int(
            fixed["smooth_secondary_window"]
        )
        definition["smooth_primary_weight"] = float(
            fixed["smooth_primary_weight"]
        )
    if "daily_entry_slots" in fixed:
        definition["daily_entry_slots"] = int(fixed["daily_entry_slots"])
    if "max_daily_score_sells" in fixed:
        definition["max_daily_score_sells"] = int(
            fixed["max_daily_score_sells"]
        )
    if "sell_confirmation_days" in fixed:
        definition["sell_confirmation_days"] = int(
            fixed["sell_confirmation_days"]
        )
    if "max_hold_renewal_policy" in fixed:
        definition["max_hold_renewal_policy"] = str(
            fixed["max_hold_renewal_policy"]
        )
    if "replacement_advantage_age_bands" in fixed:
        definition["replacement_advantage_age_bands"] = [
            [int(min_age), float(value)]
            for min_age, value in fixed["replacement_advantage_age_bands"]
        ]
    return definition


def run_case(
    arrays: dict,
    score: np.ndarray,
    order: np.ndarray,
    definition: dict,
    protocol: dict,
    start: str,
    end: str,
    record_actions: bool = False,
):
    fixed_schedule = np.full(len(arrays["dates"]), 5, dtype=np.int16)
    entry_slots_schedule = None
    if "daily_entry_slots" in definition:
        entry_slots_schedule = np.full(
            len(arrays["dates"]),
            int(definition["daily_entry_slots"]),
            dtype=np.int16,
        )
    base_protocol = json.loads(
        (
            ROOT
            / "quant/main/strategy_library/production/"
            "prod_v260_10d_regime_warmup_all4key_v20260724/"
            "inputs/preregistered_protocol.json"
        ).read_text(encoding="utf-8")
    )
    if (
        "max_daily_score_sells" in definition
        or "sell_confirmation_days" in definition
        or "max_hold_renewal_policy" in definition
        or "replacement_advantage_age_bands" in definition
        or "sell_score_mode" in definition
        or "sell_score_regime_offsets" in definition
        or "sell_score_quantile_blend" in definition
        or "sell_score_quantile_adaptive" in definition
        or "replacement_advantage_dispersion_adaptive" in definition
    ):
        v252 = v260.v258.v252
        sell_score_mode = str(definition.get("sell_score_mode", "absolute_and_relative"))
        if sell_score_mode not in {"absolute_and_relative", "relative_only"}:
            raise ValueError(f"unknown sell_score_mode: {sell_score_mode}")
        sell_score_schedule = None
        replacement_advantage_schedule = None
        if "sell_score_regime_offsets" in definition:
            weak_offset, strong_offset = definition["sell_score_regime_offsets"]
            momentum = v134.market_momentum(
                arrays, int(definition["market_lookback"])
            )
            sell_score_schedule = np.full(
                len(arrays["dates"]),
                float(definition["sell_score_below"]),
                dtype=float,
            )
            weak = np.isfinite(momentum) & (
                momentum < float(definition["market_return_min"])
            )
            strong = np.isfinite(momentum) & (
                momentum >= float(definition["strong_return_min"])
            )
            sell_score_schedule[weak] += float(weak_offset)
            sell_score_schedule[strong] += float(strong_offset)
        if "sell_score_quantile_blend" in definition:
            quantile, blend_weight = definition["sell_score_quantile_blend"]
            if not (0.0 <= quantile <= 1.0 and 0.0 <= blend_weight <= 1.0):
                raise ValueError(
                    "sell_score_quantile_blend requires quantile and weight in [0, 1]"
                )
            base_threshold = float(definition["sell_score_below"])
            sell_score_schedule = np.full(
                len(arrays["dates"]), base_threshold, dtype=float
            )
            for t in range(len(arrays["dates"])):
                valid = arrays["signal_clean"][t] & np.isfinite(score[t])
                if not np.any(valid):
                    continue
                daily_threshold = float(np.quantile(score[t, valid], quantile))
                sell_score_schedule[t] = (
                    (1.0 - blend_weight) * base_threshold
                    + blend_weight * daily_threshold
                )
        if "sell_score_quantile_adaptive" in definition:
            quantile, blend_weight, lower, upper, smooth_window = definition[
                "sell_score_quantile_adaptive"
            ]
            if not (
                0.0 <= quantile <= 1.0
                and 0.0 <= blend_weight <= 1.0
                and lower <= upper
                and smooth_window >= 1
            ):
                raise ValueError(
                    "sell_score_quantile_adaptive requires valid "
                    "quantile, weight, bounds and window"
                )
            base_threshold = float(definition["sell_score_below"])
            raw_thresholds = np.full(len(arrays["dates"]), np.nan, dtype=float)
            for t in range(len(arrays["dates"])):
                valid = arrays["signal_clean"][t] & np.isfinite(score[t])
                if np.any(valid):
                    raw_thresholds[t] = np.clip(
                        float(np.quantile(score[t, valid], quantile)),
                        lower,
                        upper,
                    )
            sell_score_schedule = np.full(
                len(arrays["dates"]), base_threshold, dtype=float
            )
            window = int(smooth_window)
            for t in range(len(arrays["dates"])):
                history = raw_thresholds[max(0, t - window + 1) : t + 1]
                history = history[np.isfinite(history)]
                if history.size:
                    adaptive_threshold = float(np.median(history))
                    sell_score_schedule[t] = (
                        (1.0 - blend_weight) * base_threshold
                        + blend_weight * adaptive_threshold
                    )
        if "replacement_advantage_dispersion_adaptive" in definition:
            (
                lower_quantile,
                upper_quantile,
                scale,
                blend_weight,
                lower,
                upper,
                smooth_window,
            ) = definition["replacement_advantage_dispersion_adaptive"]
            if not (
                0.0 <= lower_quantile < upper_quantile <= 1.0
                and scale > 0.0
                and 0.0 <= blend_weight <= 1.0
                and lower <= upper
                and smooth_window >= 1
            ):
                raise ValueError(
                    "replacement_advantage_dispersion_adaptive requires "
                    "valid quantiles, scale, weight, bounds and window"
                )
            base_advantage = float(definition["replacement_advantage"])
            raw_advantages = np.full(len(arrays["dates"]), np.nan, dtype=float)
            for t in range(len(arrays["dates"])):
                valid = arrays["signal_clean"][t] & np.isfinite(score[t])
                if not np.any(valid):
                    continue
                lower_score, upper_score = np.quantile(
                    score[t, valid], [lower_quantile, upper_quantile]
                )
                raw_advantages[t] = np.clip(
                    float(upper_score - lower_score) * scale,
                    lower,
                    upper,
                )
            replacement_advantage_schedule = np.full(
                len(arrays["dates"]), base_advantage, dtype=float
            )
            window = int(smooth_window)
            for t in range(len(arrays["dates"])):
                history = raw_advantages[max(0, t - window + 1) : t + 1]
                history = history[np.isfinite(history)]
                if history.size:
                    adaptive_advantage = float(np.median(history))
                    replacement_advantage_schedule[t] = (
                        (1.0 - blend_weight) * base_advantage
                        + blend_weight * adaptive_advantage
                    )
        return v252.v162.run_case(
            arrays,
            score,
            order,
            definition,
            base_protocol,
            end,
            start,
            record_actions=record_actions,
            selection_mask_override=v252.v174.selection_mask(
                arrays, definition["max_rank_deterioration"]
            ),
            candidate_target_multiplier_override=v252.warmup_multiplier(
                arrays, score, definition, base_protocol, start
            ),
            sell_confirmation_days_override=int(
                definition.get("sell_confirmation_days", 1)
            ),
            max_positions_override=int(definition["max_positions"]),
            max_positions_schedule_override=fixed_schedule,
            entry_slots_schedule_override=entry_slots_schedule,
            max_daily_score_sells_override=definition.get(
                "max_daily_score_sells"
            ),
            sell_score_below_override=(
                np.full(len(arrays["dates"]), np.inf, dtype=float)
                if sell_score_mode == "relative_only"
                else sell_score_schedule
            ),
            replacement_advantage_override=replacement_advantage_schedule,
            replacement_advantage_age_bands_override=definition.get(
                "replacement_advantage_age_bands"
            ),
            max_hold_renewal_policy_override=str(
                definition.get("max_hold_renewal_policy", "none")
            ),
        )
    return v260.v258.v252.run_case(
        arrays,
        score,
        order,
        definition,
        base_protocol,
        end,
        start=start,
        record_actions=record_actions,
        entry_slots_schedule_override=entry_slots_schedule,
        max_positions_schedule_override=fixed_schedule,
    )


def metric_row(
    daily: pd.DataFrame,
    start: str,
    end: str,
    prefix: str = "",
) -> dict:
    return {
        f"{prefix}{key}": value
        for key, value in core.metrics(daily, start, end).items()
    }


def observation() -> None:
    arrays, protocol, score, order = load_inputs()
    grid = protocol["观察期网格"]
    combinations = list(
        product(
            grid["常态单票目标仓位"],
            grid["弱势单票目标仓位"],
            grid["强势单票目标仓位"],
            grid["卖出分数阈值"],
            grid["替换分数优势"],
            grid["最短持仓策略"],
            grid.get("最大持仓交易日", [20]),
        )
    )
    rows: list[dict] = []
    definitions: dict[str, dict] = {}
    for values in combinations:
        definition = definition_for(protocol, values)
        case_id = stable_id(definition)
        definitions[case_id] = definition
        daily = run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            protocol["观察期"]["开始"],
            protocol["观察期"]["结束"],
        )
        row = {
            "case_id": case_id,
            "normal_target_pct": definition["normal_target_pct"],
            "weak_target_pct": definition["weak_target_pct"],
            "strong_target_pct": definition["strong_target_pct"],
            "sell_score_below": definition["sell_score_below"],
            "replacement_advantage": definition["replacement_advantage"],
            "min_hold_policy": definition["min_hold_policy"],
            "max_hold_days": definition["max_hold_days"],
            **metric_row(
                daily,
                protocol["观察期"]["开始"],
                protocol["观察期"]["结束"],
                "obs_",
            ),
        }
        slice_values = []
        slice_sharpes = []
        slice_drawdowns = []
        for item in protocol["观察期"]["分段"]:
            sliced = run_case(
                arrays,
                score,
                order,
                definition,
                protocol,
                item["开始"],
                item["结束"],
            )
            metrics = core.metrics(sliced, item["开始"], item["结束"])
            slice_values.append(float(metrics["linear_annual_proxy"]))
            slice_sharpes.append(float(metrics["sharpe"]))
            slice_drawdowns.append(float(metrics["max_drawdown"]))
        row.update(
            {
                "slice_min_linear_annual": min(slice_values),
                "slice_median_linear_annual": float(np.median(slice_values)),
                "slice_min_sharpe": min(slice_sharpes),
                "slice_max_drawdown": max(slice_drawdowns),
            }
        )
        startup_points = protocol.get("观察期冷启动点", [])
        if startup_points:
            startup_annuals = []
            startup_sharpes = []
            startup_drawdowns = []
            for start_point in startup_points:
                startup_daily = run_case(
                    arrays,
                    score,
                    order,
                    definition,
                    protocol,
                    str(start_point),
                    protocol["观察期"]["结束"],
                )
                startup_metrics = core.metrics(
                    startup_daily,
                    str(start_point),
                    protocol["观察期"]["结束"],
                )
                startup_annuals.append(
                    float(startup_metrics["linear_annual_proxy"])
                )
                startup_sharpes.append(float(startup_metrics["sharpe"]))
                startup_drawdowns.append(
                    float(startup_metrics["max_drawdown"])
                )
            row.update(
                {
                    "startup_min_linear_annual": min(startup_annuals),
                    "startup_median_linear_annual": float(
                        np.median(startup_annuals)
                    ),
                    "startup_min_sharpe": min(startup_sharpes),
                    "startup_max_drawdown": max(startup_drawdowns),
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    # 观察期选择兼顾收益、Sharpe、回撤和分段下限，避免只取尖峰。
    frame["robust_score"] = (
        frame["obs_linear_annual_proxy"]
        + 0.30 * frame["slice_median_linear_annual"]
        + 0.15 * frame["slice_min_linear_annual"]
        + 0.08 * frame["obs_sharpe"]
        + 0.04 * frame["slice_min_sharpe"]
        - 0.40 * frame["obs_max_drawdown"]
        - 0.20 * frame["slice_max_drawdown"]
    )
    if "startup_min_linear_annual" in frame:
        frame["robust_score"] += (
            0.45 * frame["startup_median_linear_annual"]
            + 0.30 * frame["startup_min_linear_annual"]
            + 0.08 * frame["startup_min_sharpe"]
            - 0.25 * frame["startup_max_drawdown"]
        )
    gate = protocol["观察期晋级门槛"]
    frame["eligible"] = (
        (frame["obs_linear_annual_proxy"] >= gate["最低线性年化"])
        & (frame["obs_sharpe"] >= gate["最低夏普"])
        & (frame["obs_max_drawdown"] <= gate["最大回撤"])
        & (frame["slice_min_linear_annual"] >= gate["分段最低线性年化"])
        & (frame["slice_min_sharpe"] >= gate["分段最低夏普"])
    )
    if "startup_min_linear_annual" in frame:
        frame["eligible"] &= (
            (
                frame["startup_min_linear_annual"]
                >= gate["冷启动最低线性年化"]
            )
            & (
                frame["startup_min_sharpe"]
                >= gate["冷启动最低夏普"]
            )
            & (
                frame["startup_max_drawdown"]
                <= gate["冷启动最大回撤"]
            )
        )
    frame = frame.sort_values(
        ["eligible", "robust_score", "obs_sharpe"],
        ascending=[False, False, False],
        kind="stable",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "观察期网格结果.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame["eligible"]].head(
        int(protocol["冻结候选数量"])
    )
    if selected.empty:
        selected = frame.head(int(protocol["冻结候选数量"]))
    frozen = {
        "状态": "观察期已完成_候选已冻结_验证期尚未读取",
        "验证期用于选择": False,
        "输入最大日期": str(arrays["dates"][-1]),
        "网格数量": len(frame),
        "符合观察期门槛数量": int(frame["eligible"].sum()),
        "候选": [
            {
                "case_id": str(row.case_id),
                "definition": definitions[str(row.case_id)],
                "observation_metrics": row.to_dict(),
            }
            for _, row in selected.iterrows()
        ],
    }
    FROZEN.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "网格数量": len(frame),
                "符合门槛": int(frame["eligible"].sum()),
                "冻结候选": len(selected),
                "冻结文件": str(FROZEN),
            },
            ensure_ascii=False,
        )
    )


def validation() -> None:
    arrays, protocol, score, order = load_inputs()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in frozen["候选"]:
        definition = item["definition"]
        daily, actions = run_case(
            arrays,
            score,
            order,
            definition,
            protocol,
            protocol["验证期"]["开始"],
            protocol["验证期"]["结束"],
            record_actions=True,
        )
        rows.append(
            {
                "case_id": item["case_id"],
                **metric_row(
                    daily,
                    protocol["验证期"]["开始"],
                    protocol["验证期"]["结束"],
                    "val_",
                ),
                "action_rows": len(actions),
                "buy_rows": int((actions["action"] == "BUY").sum()),
                "sell_rows": int((actions["action"] == "SELL").sum()),
                "bj_rows": int(
                    actions["stock_code"]
                    .astype(str)
                    .str.endswith(".BJ")
                    .sum()
                ),
                "duplicate_actions": int(
                    actions.duplicated(
                        ["signal_date", "buy_date", "action", "stock_code"]
                    ).sum()
                ),
            }
        )
        actions.to_csv(
            OUT / f"{item['case_id']}_验证期动作.csv",
            index=False,
            encoding="utf-8-sig",
        )
        daily.to_csv(
            OUT / f"{item['case_id']}_验证期净值.csv",
            index=False,
            encoding="utf-8-sig",
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "冻结候选验证期结果.csv", index=False, encoding="utf-8-sig")
    print(result.to_json(orient="records", force_ascii=False))


def export_frozen_best() -> None:
    arrays, protocol, score, order = load_inputs()
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    item = frozen["候选"][0]
    outputs = {}
    for label, window in (
        ("观察期", protocol["观察期"]),
        ("验证期", protocol["验证期"]),
    ):
        daily, actions = run_case(
            arrays,
            score,
            order,
            item["definition"],
            protocol,
            window["开始"],
            window["结束"],
            record_actions=True,
        )
        daily_path = OUT / f"{item['case_id']}_{label}净值.csv"
        action_path = OUT / f"{item['case_id']}_{label}动作.csv"
        daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
        actions.to_csv(action_path, index=False, encoding="utf-8-sig")
        outputs[label] = {
            "开始": window["开始"],
            "结束": window["结束"],
            "净值文件": str(daily_path),
            "动作文件": str(action_path),
            "动作数": len(actions),
            "买入数": int((actions["action"] == "BUY").sum()),
            "卖出数": int((actions["action"] == "SELL").sum()),
        }
    manifest = {
        "状态": "research_only_frozen_best_exported",
        "case_id": item["case_id"],
        "definition": item["definition"],
        "outputs": outputs,
    }
    (OUT / "冻结最佳候选导出清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


def startup_stability() -> None:
    arrays, protocol, score, order = load_inputs()
    stability_protocol = json.loads(
        (OUT / "稳定性检查协议.json").read_text(encoding="utf-8")
    )
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    item = frozen["候选"][0]
    definition = item["definition"]
    full_daily = run_case(
        arrays,
        score,
        order,
        definition,
        protocol,
        protocol["观察期"]["开始"],
        protocol["观察期"]["结束"],
    )
    rows = []
    for group in stability_protocol["启动点检查"]:
        for start in group["起点"]:
            cold = run_case(
                arrays,
                score,
                order,
                definition,
                protocol,
                start,
                protocol["观察期"]["结束"],
            )
            cold_metrics = core.metrics(
                cold, start, protocol["观察期"]["结束"]
            )
            continuous_metrics = core.metrics(
                full_daily, start, protocol["观察期"]["结束"]
            )
            rows.append(
                {
                    "case_id": item["case_id"],
                    "anchor": group["名称"],
                    "start": start,
                    **{
                        f"cold_{key}": value
                        for key, value in cold_metrics.items()
                    },
                    **{
                        f"continuous_{key}": value
                        for key, value in continuous_metrics.items()
                    },
                    "annual_gap": (
                        float(cold_metrics["linear_annual_proxy"])
                        - float(
                            continuous_metrics["linear_annual_proxy"]
                        )
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "相邻启动点稳定性.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby("anchor")
        .agg(
            starts=("start", "count"),
            cold_min_annual=("cold_linear_annual_proxy", "min"),
            cold_median_annual=("cold_linear_annual_proxy", "median"),
            cold_max_annual=("cold_linear_annual_proxy", "max"),
            cold_min_sharpe=("cold_sharpe", "min"),
            cold_max_drawdown=("cold_max_drawdown", "max"),
            max_abs_annual_gap=("annual_gap", lambda x: x.abs().max()),
        )
        .reset_index()
    )
    summary.to_csv(
        OUT / "相邻启动点稳定性汇总.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_json(orient="records", force_ascii=False))


def contribution_stress() -> None:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    item = frozen["候选"][0]
    daily = pd.read_csv(
        OUT / f"{item['case_id']}_观察期净值.csv",
        dtype={"date": str},
    )
    daily["date"] = daily["date"].astype(str).str.replace("-", "")
    daily["return"] = pd.to_numeric(daily["return"], errors="coerce").fillna(0.0)
    daily["month"] = daily["date"].str[:6]
    monthly = (
        daily.groupby("month")["return"]
        .apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
        .reset_index(name="return")
        .sort_values("return", ascending=False)
    )
    monthly.to_csv(
        OUT / "观察期月度收益.csv", index=False, encoding="utf-8-sig"
    )
    base_cumulative = float(np.prod(1.0 + daily["return"]) - 1.0)
    top_day_index = daily["return"].idxmax()
    without_best_day = float(
        np.prod(1.0 + daily.drop(index=top_day_index)["return"]) - 1.0
    )
    top_count = max(int(np.ceil(len(daily) * 0.01)), 1)
    without_top1pct = float(
        np.prod(
            1.0
            + daily.drop(
                index=daily.nlargest(top_count, "return").index
            )["return"]
        )
        - 1.0
    )
    best_month = str(monthly.iloc[0]["month"])
    without_best_month = float(
        np.prod(1.0 + daily[daily["month"] != best_month]["return"]) - 1.0
    )
    result = {
        "case_id": item["case_id"],
        "daily_rows": len(daily),
        "base_cumulative_return": base_cumulative,
        "best_day": str(daily.loc[top_day_index, "date"]),
        "best_day_return": float(daily.loc[top_day_index, "return"]),
        "without_best_day_cumulative_return": without_best_day,
        "top_1pct_day_count": top_count,
        "without_top_1pct_days_cumulative_return": without_top1pct,
        "best_month": best_month,
        "best_month_return": float(monthly.iloc[0]["return"]),
        "without_best_month_cumulative_return": without_best_month,
    }
    (OUT / "收益贡献压力测试.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="V260 五只持仓观察期与验证期研究")
    parser.add_argument(
        "stage",
        choices=[
            "observation",
            "validation",
            "export",
            "startup",
            "stress",
        ],
        help="必须先运行 observation 冻结候选，再单独运行 validation",
    )
    args = parser.parse_args()
    if args.stage == "observation":
        observation()
    elif args.stage == "validation":
        validation()
    elif args.stage == "export":
        export_frozen_best()
    elif args.stage == "startup":
        startup_stability()
    else:
        contribution_stress()


if __name__ == "__main__":
    main()

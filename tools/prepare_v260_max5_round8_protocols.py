from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "quant" / "data_file" / "reports"
SOURCE_DIR = (
    REPORTS / "strategy_agent_v260_max5_observation_validation_20260727_round5"
)


def source_protocol_path() -> Path:
    preferred = SOURCE_DIR / "预注册协议.json"
    if preferred.exists():
        return preferred
    matches = []
    for path in SOURCE_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        try:
            observation_grid(payload)
        except RuntimeError:
            continue
        matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"无法唯一识别源预注册协议：{matches}")
    return matches[0]


def fixed_parameters(protocol: dict) -> dict:
    matches = [
        value
        for value in protocol.values()
        if isinstance(value, dict) and 700000 in value.values()
    ]
    if len(matches) != 1:
        raise RuntimeError("无法唯一识别固定参数区")
    return matches[0]


def set_current_weight(fixed: dict, weight: float) -> None:
    keys = [
        key
        for key, value in fixed.items()
        if isinstance(value, float) and abs(value - 0.1) < 1e-12
    ]
    if len(keys) != 1:
        raise RuntimeError("无法唯一识别当日权重字段")
    fixed[keys[0]] = weight


def set_smooth_window(fixed: dict, window: int) -> None:
    keys = [
        key
        for key, value in fixed.items()
        if isinstance(value, int) and not isinstance(value, bool) and value == 7
    ]
    if len(keys) != 1:
        raise RuntimeError("无法唯一识别平滑窗口字段")
    fixed[keys[0]] = window


def observation_grid(protocol: dict) -> dict:
    matches = [
        value
        for value in protocol.values()
        if isinstance(value, dict)
        and any(
            isinstance(item, list) and "normal_3_weak_4" in item
            for item in value.values()
        )
    ]
    if len(matches) != 1:
        raise RuntimeError("无法唯一识别观察期网格")
    return matches[0]


def replace_grid_value(grid: dict, old: list, new: list) -> None:
    keys = [key for key, value in grid.items() if value == old]
    if len(keys) != 1:
        raise RuntimeError(f"无法唯一识别网格字段：{old}")
    grid[keys[0]] = new


def main() -> None:
    source = json.loads(source_protocol_path().read_text(encoding="utf-8"))
    for suffix, weight in (("round8c05", 0.05), ("round8c15", 0.15)):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, weight)
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: current_weight={weight}, path={output}")

    for suffix, weight in (
        ("round18c11", 0.11),
        ("round18c13", 0.13),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, weight)
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: current_weight={weight}, path={output}")

    for suffix, weight in (
        ("round19c115", 0.115),
        ("round19c125", 0.125),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, weight)
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: current_weight={weight}, path={output}")

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.29, 0.31])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round9pos"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / "预注册协议.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"round9pos: normal_target=[0.29,0.31], path={output}")

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.86, 0.88])
    replace_grid_value(grid, [0.05, 0.07], [0.06, 0.08])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round10exit"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / "预注册协议.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round10exit: sell_score=[0.86,0.88], "
        f"replacement_advantage=[0.06,0.08], path={output}"
    )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05, 0.07])
    replace_grid_value(grid, [0.22, 0.25], [0.21, 0.23])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round11regime"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / "预注册协议.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round11regime: weak_target=[0.05,0.07], "
        f"strong_target=[0.21,0.23], path={output}"
    )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [18, 20, 22], [23, 24])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round12hold"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / "预注册协议.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"round12hold: max_hold_days=[23,24], path={output}")

    for suffix, threshold in (
        ("round13strong30", 0.03),
        ("round13strong40", 0.04),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        fixed["强势收益阈值"] = threshold
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: strong_return_min={threshold}, path={output}"
        )

    for suffix, lookback in (
        ("round14market8", 8),
        ("round14market12", 12),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        fixed["市场观察交易日"] = lookback
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: market_lookback={lookback}, path={output}")

    for suffix, threshold in (
        ("round15marketm005", -0.005),
        ("round15marketp005", 0.005),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        fixed["市场收益分界"] = threshold
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: market_return_min={threshold}, path={output}")

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(
        grid,
        ["normal_3_weak_4"],
        [
            "fixed_4",
            "strong_3_else_4",
            "weak_3_else_4",
            "strong_3_normal_4_weak_5",
        ],
    )
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round16minhold"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / "预注册协议.json"
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("round16minhold: four legal min-hold policies, path={}".format(output))

    for suffix, weight in (
        ("round17c08", 0.08),
        ("round17c12", 0.12),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, weight)
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / "预注册协议.json"
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{suffix}: current_weight={weight}, path={output}")


    for suffix, threshold in (
        ("round20d08", 0.08),
        ("round20d12", 0.12),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = threshold
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            f"max_rank_deterioration={threshold}, path={output}"
        )

    for suffix, low_gross in (
        ("round21low18", 0.18),
        ("round21low22", 0.22),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["low_gross"] = low_gross
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"low_gross={low_gross}, path={output}"
        )

    for suffix, threshold in (
        ("round22d04", 0.04),
        ("round22d06", 0.06),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = threshold
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            f"max_rank_deterioration={threshold}, path={output}"
        )

    for suffix, smooth_window in (
        ("round23smooth5", 5),
        ("round23smooth6", 6),
        ("round24smooth8", 8),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        set_smooth_window(fixed, smooth_window)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"smooth_window={smooth_window}, path={output}"
        )

    for suffix, weight_7d in (
        ("round25ensemble50", 0.50),
        ("round25ensemble75", 0.75),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["smooth_7d_ensemble_weight"] = weight_7d
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"smooth_7d_ensemble_weight={weight_7d}, path={output}"
        )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    fixed = fixed_parameters(protocol)
    set_current_weight(fixed, 0.12)
    fixed["max_rank_deterioration"] = 0.08
    fixed["daily_entry_slots"] = 2
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round26entry2"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / source_protocol_path().name
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round26entry2: current_weight=0.12, "
        "max_rank_deterioration=0.08, daily_entry_slots=2, "
        f"path={output}"
    )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    fixed = fixed_parameters(protocol)
    set_current_weight(fixed, 0.12)
    fixed["max_rank_deterioration"] = 0.08
    fixed["max_daily_score_sells"] = 1
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round27sellcap1"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / source_protocol_path().name
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round27sellcap1: current_weight=0.12, "
        "max_rank_deterioration=0.08, max_daily_score_sells=1, "
        f"path={output}"
    )


    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    fixed = fixed_parameters(protocol)
    set_current_weight(fixed, 0.12)
    fixed["max_rank_deterioration"] = 0.08
    fixed["sell_confirmation_days"] = 2
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.07])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_round28confirm2"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / source_protocol_path().name
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round28confirm2: current_weight=0.12, "
        "max_rank_deterioration=0.08, sell_confirmation_days=2, "
        f"path={output}"
    )

    for suffix, weak_target, strong_target in (
        ("round29strong23", 0.06, 0.23),
        ("round29weak07", 0.07, 0.22),
        ("round30strong21", 0.06, 0.21),
        ("round30weak05", 0.05, 0.22),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [weak_target])
        replace_grid_value(grid, [0.22, 0.25], [strong_target])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"weak_target={weak_target}, strong_target={strong_target}, "
            f"path={output}"
        )

    for suffix, normal_target in (
        ("round31normal29", 0.29),
        ("round31normal31", 0.31),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [normal_target])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"normal_target={normal_target}, path={output}"
        )

    for suffix, max_hold_days in (
        ("round32hold21", 21),
        ("round32hold23", 23),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.07])
        replace_grid_value(grid, [18, 20, 22], [max_hold_days])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"max_hold_days={max_hold_days}, path={output}"
        )

    for suffix, sell_score, replacement_advantage in (
        ("round33sell865", 0.865, 0.07),
        ("round33sell875", 0.875, 0.07),
        ("round33replace065", 0.87, 0.065),
        ("round33replace075", 0.87, 0.075),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [sell_score])
        replace_grid_value(grid, [0.05, 0.07], [replacement_advantage])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, "
            f"sell_score={sell_score}, "
            f"replacement_advantage={replacement_advantage}, path={output}"
        )

    for suffix, max_hold_days in (
        ("round34replace065hold21", 21),
        ("round34replace065hold23", 23),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [max_hold_days])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"max_hold_days={max_hold_days}, path={output}"
        )

    for suffix, renewal_policy in (
        ("round35renewtop1", "top1"),
        ("round35renewnoscoreexit", "no_score_exit"),
        ("round39renewtop1rebalance", "top1_rebalance"),
        ("round39renewnoscoreexitrebalance", "no_score_exit_rebalance"),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["max_hold_renewal_policy"] = renewal_policy
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"max_hold_renewal_policy={renewal_policy}, path={output}"
        )

    for suffix, age_bands in (
        ("round36age18a04", [[18, 0.04]]),
        ("round36age18a05_age20a03", [[18, 0.05], [20, 0.03]]),
        ("round37age17a05_age19a03", [[17, 0.05], [19, 0.03]]),
        ("round37age19a05_age21a03", [[19, 0.05], [21, 0.03]]),
        ("round38age18a055_age20a035", [[18, 0.055], [20, 0.035]]),
        ("round38age18a045_age20a025", [[18, 0.045], [20, 0.025]]),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = age_bands
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"replacement_advantage_age_bands={age_bands}, path={output}"
        )

    for suffix, min_hold_policy in (
        ("round40minholdfixed4", "fixed_4"),
        ("round40minholdstrong3else4", "strong_3_else_4"),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, ["normal_3_weak_4"], [min_hold_policy])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"min_hold_policy={min_hold_policy}, path={output}"
        )

    for suffix, strong_return_min in (
        ("round41strong0325", 0.0325),
        ("round41strong0375", 0.0375),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        strong_keys = ["strong_return_min"]
        if len(strong_keys) != 1:
            raise RuntimeError("无法唯一识别强势收益阈值字段")
        fixed[strong_keys[0]] = strong_return_min
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"strong_return_min={strong_return_min}, path={output}"
        )

    for suffix, market_lookback in (
        ("round42market9", 9),
        ("round42market11", 11),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["market_lookback"] = market_lookback
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"market_lookback={market_lookback}, path={output}"
        )

    for suffix, market_return_min in (
        ("round43marketm0025", -0.0025),
        ("round43marketp0025", 0.0025),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["market_return_min"] = market_return_min
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"market_return_min={market_return_min}, path={output}"
        )

    for suffix, weak_target in (
        ("round44weak05agebands", 0.05),
        ("round44weak055agebands", 0.055),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [weak_target])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"weak_target={weak_target}, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, normal_target in (
        ("round45normal29weak05agebands", 0.29),
        ("round45normal295weak05agebands", 0.295),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [normal_target])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"normal_target={normal_target}, weak_target=0.05, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, strong_target in (
        ("round46strong225weak05agebands", 0.225),
        ("round46strong23weak05agebands", 0.23),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [strong_target])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, replacement_advantage=0.065, "
            f"normal_target=0.3, weak_target=0.05, strong_target={strong_target}, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, replacement_advantage in (
        ("round47replace060weak05agebands", 0.06),
        ("round47replace0625weak05agebands", 0.0625),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [replacement_advantage])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, normal_target=0.3, weak_target=0.05, "
            f"strong_target=0.22, replacement_advantage={replacement_advantage}, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, sell_score in (
        ("round48sell8675weak05agebands", 0.8675),
        ("round48sell8725weak05agebands", 0.8725),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [sell_score])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, normal_target=0.3, weak_target=0.05, "
            f"strong_target=0.22, sell_score={sell_score}, "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    fixed = fixed_parameters(protocol)
    set_current_weight(fixed, 0.12)
    fixed["max_rank_deterioration"] = 0.08
    fixed["sell_score_mode"] = "relative_only"
    fixed["replacement_advantage_age_bands"] = [
        [18, 0.05],
        [20, 0.03],
    ]
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.065, 0.08])
    replace_grid_value(grid, [18, 20, 22], [22])
    target = (
        REPORTS
        / "strategy_agent_v260_max5_observation_validation_20260727_"
        "round49relativeonly"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / source_protocol_path().name
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "round49relativeonly: current_weight=0.12, "
        "max_rank_deterioration=0.08, normal_target=0.3, weak_target=0.05, "
        "strong_target=0.22, sell_score_mode=relative_only, "
        "replacement_advantage=[0.065,0.08], "
        "age_bands=[[18,0.05],[20,0.03]], "
        f"path={output}"
    )

    for suffix, offset in (
        ("round50regimesell00125", 0.00125),
        ("round50regimesell0025", 0.0025),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_regime_offsets"] = [offset, -offset]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, "
            "max_rank_deterioration=0.08, normal_target=0.3, weak_target=0.05, "
            "strong_target=0.22, sell_score=0.87, "
            f"sell_score_regime_offsets=[{offset},{-offset}], "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, blend_weight in (
        ("round51quantile87blend25", 0.25),
        ("round51quantile87blend50", 0.50),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_blend"] = [0.87, blend_weight]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, max_rank_deterioration=0.08, "
            "normal_target=0.3, weak_target=0.05, strong_target=0.22, "
            f"sell_score_quantile_blend=[0.87,{blend_weight}], "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, blend_weight in (
        ("round52quantile90blend25", 0.25),
        ("round52quantile90blend50", 0.50),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_blend"] = [0.90, blend_weight]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, max_rank_deterioration=0.08, "
            "normal_target=0.3, weak_target=0.05, strong_target=0.22, "
            f"sell_score_quantile_blend=[0.90,{blend_weight}], "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, blend_weight in (
        ("round53adaptiveq90w10", 0.10),
        ("round53adaptiveq90w20", 0.20),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            blend_weight,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, max_rank_deterioration=0.08, "
            "normal_target=0.3, weak_target=0.05, strong_target=0.22, "
            "sell_score_quantile_adaptive="
            f"[0.90,{blend_weight},0.8675,0.8725,10], "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, quantile, smooth_window in (
        ("round54adaptiveq89w10", 0.89, 10),
        ("round54adaptiveq91w10", 0.91, 10),
        ("round54adaptiveq90win5", 0.90, 5),
        ("round54adaptiveq90win20", 0.90, 20),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            quantile,
            0.10,
            0.8675,
            0.8725,
            smooth_window,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight=0.12, max_rank_deterioration=0.08, "
            "normal_target=0.3, weak_target=0.05, strong_target=0.22, "
            "sell_score_quantile_adaptive="
            f"[{quantile},0.10,0.8675,0.8725,{smooth_window}], "
            "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    protocol = json.loads(json.dumps(source, ensure_ascii=False))
    fixed = fixed_parameters(protocol)
    set_current_weight(fixed, 0.12)
    fixed["max_rank_deterioration"] = 0.08
    fixed["sell_score_quantile_adaptive"] = [
        0.90,
        0.10,
        0.8675,
        0.8725,
        10,
    ]
    fixed["replacement_advantage_age_bands"] = [
        [18, 0.05],
        [20, 0.03],
    ]
    grid = observation_grid(protocol)
    replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
    replace_grid_value(grid, [0.06, 0.08, 0.1], [0.06])
    replace_grid_value(grid, [0.22, 0.25], [0.22])
    replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
    replace_grid_value(grid, [0.05, 0.07], [0.065])
    replace_grid_value(grid, [18, 20, 22], [22])
    suffix = "round55adaptiveq90weak06"
    target = (
        REPORTS
        / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
    )
    target.mkdir(parents=True, exist_ok=True)
    output = target / source_protocol_path().name
    output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{suffix}: current_weight=0.12, max_rank_deterioration=0.08, "
        "normal_target=0.3, weak_target=0.06, strong_target=0.22, "
        "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
        "replacement_advantage=0.065, age_bands=[[18,0.05],[20,0.03]], "
        f"path={output}"
    )

    for suffix, blend_weight in (
        ("round56adaptivegapw10", 0.10),
        ("round56adaptivegapw20", 0.20),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_dispersion_adaptive"] = [
            0.90,
            0.95,
            1.15,
            blend_weight,
            0.055,
            0.075,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: weak_target=0.05, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage_dispersion_adaptive="
            f"[0.90,0.95,1.15,{blend_weight},0.055,0.075,10], "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, normal_target in (
        ("round57normal305adaptive", 0.305),
        ("round57normal310adaptive", 0.310),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [normal_target])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: normal_target={normal_target}, weak_target=0.05, "
            "strong_target=0.22, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage=0.065, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, current_weight in (
        ("round58current115adaptive", 0.115),
        ("round58current125adaptive", 0.125),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, current_weight)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: current_weight={current_weight}, "
            "normal_target=0.30, weak_target=0.05, strong_target=0.22, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage=0.065, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, score_window in (
        ("round59scorewin6adaptive", 6),
        ("round59scorewin8adaptive", 8),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        set_smooth_window(fixed, score_window)
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: score_window={score_window}, current_weight=0.12, "
            "normal_target=0.30, weak_target=0.05, strong_target=0.22, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage=0.065, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, weight_7d in (
        ("round60ensemble7d90adaptive", 0.90),
        ("round60ensemble7d75adaptive", 0.75),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["smooth_7d_ensemble_weight"] = weight_7d
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: smooth_7d_ensemble_weight={weight_7d}, "
            "current_weight=0.12, normal_target=0.30, weak_target=0.05, "
            "strong_target=0.22, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage=0.065, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )

    for suffix, primary_weight in (
        ("round61ensemble7d8d90adaptive", 0.90),
        ("round61ensemble7d8d75adaptive", 0.75),
    ):
        protocol = json.loads(json.dumps(source, ensure_ascii=False))
        fixed = fixed_parameters(protocol)
        set_current_weight(fixed, 0.12)
        fixed["smooth_secondary_window"] = 8
        fixed["smooth_primary_weight"] = primary_weight
        fixed["max_rank_deterioration"] = 0.08
        fixed["sell_score_quantile_adaptive"] = [
            0.90,
            0.10,
            0.8675,
            0.8725,
            10,
        ]
        fixed["replacement_advantage_age_bands"] = [
            [18, 0.05],
            [20, 0.03],
        ]
        grid = observation_grid(protocol)
        replace_grid_value(grid, [0.28, 0.3, 0.32], [0.3])
        replace_grid_value(grid, [0.06, 0.08, 0.1], [0.05])
        replace_grid_value(grid, [0.22, 0.25], [0.22])
        replace_grid_value(grid, [0.85, 0.87, 0.89], [0.87])
        replace_grid_value(grid, [0.05, 0.07], [0.065])
        replace_grid_value(grid, [18, 20, 22], [22])
        target = (
            REPORTS
            / f"strategy_agent_v260_max5_observation_validation_20260727_{suffix}"
        )
        target.mkdir(parents=True, exist_ok=True)
        output = target / source_protocol_path().name
        output.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"{suffix}: primary_window=7, secondary_window=8, "
            f"primary_weight={primary_weight}, current_weight=0.12, "
            "normal_target=0.30, weak_target=0.05, strong_target=0.22, "
            "sell_score_quantile_adaptive=[0.90,0.10,0.8675,0.8725,10], "
            "replacement_advantage=0.065, "
            "age_bands=[[18,0.05],[20,0.03]], "
            f"path={output}"
        )


if __name__ == "__main__":
    main()

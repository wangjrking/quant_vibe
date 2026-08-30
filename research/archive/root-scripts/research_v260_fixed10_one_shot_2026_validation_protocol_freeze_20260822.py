from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import sys
from functools import lru_cache
from pathlib import Path


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1


REPORTS = REPO / "quant/data_file/reports"
OUTPUT_ROOT = REPORTS / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822"
CHECKPOINT = (
    REPORTS
    / "strategy_agent_v260_fixed10_residual_cash_sweep_checkpoint_20260823/"
    "pre2026_checkpoint.json"
)
EVENT_CONTRACT = (
    REPORTS
    / "strategy_agent_v260_fixed10_tushare_event_overlay_freeze_20260822/"
    "event_overlay_contract.json"
)
FROZEN_CANDIDATE = "fixed10_global_rank11_to_9_entry_buy_day_cash_sweep"
VALIDATION_RUNNER = (
    MAIN_ROOT / "research_v260_fixed10_one_shot_2026_validation_runner_20260822.py"
)
RESULT_AUDITOR = (
    MAIN_ROOT / "research_v260_fixed10_one_shot_2026_result_audit_20260822.py"
)
FROZEN_PRODUCTION_HARNESS = (
    REPO
    / "quant/data_file/runtime/agent_workspaces/strategy-agent/work"
    / "prod_v260_fixed10_equalweight_development_comparison_20260812_r2"
    / "observation_attempt_v1/run_comparison.py"
)
PRODUCTION_ROOT = (
    MAIN_ROOT
    / "strategy_library/production/prod_v260_10d_regime_warmup_all4key_v20260724"
)
CONTROL_PLANE_FILES = (
    MAIN_ROOT / "strategy_library/registry.json",
    PRODUCTION_ROOT / "trading_rules.json",
    PRODUCTION_ROOT / "inputs/preregistered_protocol.json",
    MAIN_ROOT
    / "config/prediction_manifests/executable_1d_open_return_l4_formal_20260619.json",
    MAIN_ROOT
    / "config/prediction_manifests/executable_3d_open_return_l4_formal_20260617.json",
    MAIN_ROOT
    / "config/prediction_manifests/executable_5d_open_return_l4_formal_20260620.json",
    MAIN_ROOT
    / "config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json",
)
RUNTIME_PACKAGES = ("duckdb", "numpy", "pandas")


def build_runtime_contract(checkpoint: dict) -> dict:
    if checkpoint.get("selected_candidate") != FROZEN_CANDIDATE:
        raise RuntimeError("frozen candidate identity drifted")
    policy = checkpoint.get("selected_policy", {})
    priority = policy.get("confirmed_weak_sell_priority", {})
    entry_sizing = policy.get("entry_rank_sizing", {})
    cash_sweep = policy.get("residual_cash_sweep", {})
    expected = {
        "score_sell_pressure_trigger_rule": "4_strong_market_5_weak_market",
        "score_sell_pressure_extra_min_age_rule": "4_strong_market_10_weak_market",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise RuntimeError(f"frozen candidate runtime rule drifted: {key}")
    if priority != {
        "confirmation_days": 2,
        "formula": "exit_score - 0.05 * trailing_20d_volatility_percentile",
        "changes_eligibility": False,
        "changes_exit_count": False,
    }:
        raise RuntimeError("frozen candidate sell-priority rule drifted")
    if entry_sizing != {
        "application": "new_entries_only",
        "ranking": "global_existing_frozen_score_order_before_execution_refill",
        "positions": 10,
        "top_multiplier": 1.1,
        "bottom_multiplier": 0.9,
        "schedule": "linear_descending",
        "reference_top10_multiplier_sum": 10.0,
        "refill_outside_global_top10_multiplier": 1.0,
        "actual_entry_batch_gross_neutral": False,
        "equalweight_is_soft_reference": True,
    }:
        raise RuntimeError("frozen candidate entry-rank sizing drifted")
    if cash_sweep != {
        "enabled": True,
        "trigger": "after_buy_trade",
        "recipient": "most_underweight_existing_buyable_holding",
        "target_pct": "gross_target_divided_by_max_positions",
        "board_lot_shares": 100,
        "max_orders_per_trade_day": None,
        "new_names_allowed": False,
        "forced_sales_allowed": False,
        "equalweight_and_full_investment_are_soft_directions": True,
    }:
        raise RuntimeError("frozen candidate residual-cash sweep drifted")
    return {
        "score_construction": {
            "weight_5d": 0.0,
            "smooth_window": 7,
            "current_weight": 0.1,
        },
        "pressure_regime": {
            "strong_trigger": 4,
            "weak_trigger": 5,
            "strong_extra_min_age": 4,
            "weak_extra_min_age": 10,
        },
        "confirmed_weak_sell_priority": {
            "confirmation_days": int(priority["confirmation_days"]),
            "trailing_volatility_window": 20,
            "volatility_percentile_penalty": 0.05,
            "changes_eligibility": False,
            "changes_exit_count": False,
        },
        "entry_rank_sizing": entry_sizing,
        "residual_cash_sweep": cash_sweep,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def local_module_index() -> dict[str, tuple[Path, ...]]:
    index: dict[str, set[Path]] = {}
    for path in MAIN_ROOT.rglob("*.py"):
        relative = path.relative_to(MAIN_ROOT)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            module_parts = parts[:-1]
        else:
            module_parts = parts[:-1] + [path.stem]
        for start in range(len(module_parts)):
            name = ".".join(module_parts[start:])
            index.setdefault(name, set()).add(path.resolve())
    return {
        name: tuple(sorted(paths, key=lambda item: str(item).replace("\\", "/")))
        for name, paths in index.items()
    }


def local_module_paths(module: str) -> list[Path]:
    normalized = module
    if normalized.startswith("quant.main."):
        normalized = normalized[len("quant.main.") :]
    elif normalized == "quant.main":
        return []
    return list(local_module_index().get(normalized, ()))


def module_name(path: Path) -> tuple[str, bool]:
    relative = path.resolve().relative_to(MAIN_ROOT.resolve())
    is_package = relative.name == "__init__.py"
    parts = list(relative.parts[:-1])
    if not is_package:
        parts.append(relative.stem)
    return ".".join(parts), is_package


def imported_local_paths(path: Path) -> set[Path]:
    try:
        current_module, is_package = module_name(path)
        package_parts = (
            current_module.split(".")
            if is_package
            else current_module.split(".")[:-1]
        )
    except ValueError:
        package_parts = []
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    dependencies: set[Path] = set()
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package_parts) - (node.level - 1)
                if keep < 0:
                    continue
                prefix = package_parts[:keep]
                if node.module:
                    prefix.extend(node.module.split("."))
                base = ".".join(prefix)
            else:
                base = node.module or ""
            if base:
                targets.append(base)
            targets.extend(
                f"{base}.{alias.name}" if base else alias.name
                for alias in node.names
                if alias.name != "*"
            )
        for target in targets:
            dependencies.update(local_module_paths(target))
    return dependencies


def executable_code_closure() -> list[Path]:
    if not FROZEN_PRODUCTION_HARNESS.is_file():
        raise FileNotFoundError("frozen production harness is unavailable")
    pending = [
        VALIDATION_RUNNER.resolve(),
        RESULT_AUDITOR.resolve(),
        FROZEN_PRODUCTION_HARNESS.resolve(),
    ]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"validation executable source missing: {path}")
        visited.add(path)
        pending.extend(imported_local_paths(path) - visited)
    return sorted(visited, key=lambda path: str(path).replace("\\", "/"))


def build_executable_code_binding() -> dict:
    files = [
        {
            "path": str(path).replace("\\", "/"),
            "sha256": sha256(path),
        }
        for path in executable_code_closure()
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def build_control_plane_binding() -> dict:
    missing = [path for path in CONTROL_PLANE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"control-plane source missing: {missing[0]}")
    files = [
        {
            "path": str(path.resolve()).replace("\\", "/"),
            "sha256": sha256(path),
        }
        for path in sorted(
            CONTROL_PLANE_FILES,
            key=lambda item: str(item.resolve()).replace("\\", "/"),
        )
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def build_runtime_environment_binding() -> dict:
    packages = {
        name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES
    }
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(str(value) for value in sys.version_info[:3]),
            "executable": str(Path(sys.executable).resolve()).replace("\\", "/"),
        },
        "packages": packages,
    }


def build_protocol(checkpoint: dict, event_contract: dict) -> dict:
    if checkpoint["validation_2026_opened"] or event_contract["validation_2026_opened"]:
        raise PermissionError("2026 validation was already opened")
    runtime_contract = build_runtime_contract(checkpoint)
    return {
        "status": "one_shot_2026_validation_protocol_frozen_not_executed",
        "protocol_id": "fixed10_pre2026_frozen_one_shot_2026_v3",
        "development_boundary": {
            "start": "20220101",
            "end": "20251231",
            "frozen_candidate": checkpoint["selected_candidate"],
            "candidate_policy": checkpoint["selected_policy"],
            "no_further_parameter_selection_after_validation": True,
        },
        "runtime_contract": runtime_contract,
        "validation_boundary": {
            "start": "20260105",
            "end": "20260820",
            "read_once": True,
            "development_overlap_allowed": False,
        },
        "arms": event_contract["one_shot_validation_arms"],
        "base_candidate_decision": {
            "primary": "cumulative_return_above_production",
            "machine_thresholds": {
                "deterministic_replay_required": True,
            },
            "required": {
                "cumulative_return_above_production": True,
                "deterministic_replay": True,
            },
            "reported_without_extra_gates": [
                "stress_0_65pct_cumulative_return", "cagr",
                "sharpe_delta_vs_production", "max_drawdown_delta_vs_production",
                "turnover_delta_vs_production", "annual_or_partial_year_return",
                "full_10_position_ratio", "average_invested_ratio",
            ],
        },
        "event_overlay_decision": event_contract["decision_order"]["event_overlay"],
        "event_overlay_role": event_contract["event_overlay_role"],
        "event_overlay_runtime": {
            "contract_id": event_contract["contract_id"],
            "diagnostic_coverage": event_contract["diagnostic_coverage"],
        },
        "hard_boundaries": {
            "no_parameter_tuning": True,
            "no_extra_candidate": True,
            "event_overlay_cannot_change_base_decision": True,
            "no_production_modification": True,
            "no_trading_or_network": True,
            "same_executor_and_costs_for_all_arms": True,
        },
        "source_bindings": {
            "candidate_checkpoint": str(CHECKPOINT).replace("\\", "/"),
            "candidate_checkpoint_sha256": sha256(CHECKPOINT),
            "event_overlay_contract": str(EVENT_CONTRACT).replace("\\", "/"),
            "event_overlay_contract_sha256": sha256(EVENT_CONTRACT),
            "executable_code": build_executable_code_binding(),
            "control_plane": build_control_plane_binding(),
            "runtime_environment": build_runtime_environment_binding(),
        },
        "validation_2026_opened": False,
        "production_modified": False,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    event_contract = json.loads(EVENT_CONTRACT.read_text(encoding="utf-8"))
    protocol = build_protocol(checkpoint, event_contract)
    round1.atomic_json(OUTPUT_ROOT / "one_shot_validation_protocol.json", protocol)
    print(json.dumps({
        "status": protocol["status"],
        "validation_boundary": protocol["validation_boundary"],
        "arms": protocol["arms"],
        "validation_2026_opened": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

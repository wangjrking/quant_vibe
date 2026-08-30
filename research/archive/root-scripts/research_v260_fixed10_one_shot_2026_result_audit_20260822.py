from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


REPO = Path(r"D:/work/quant/quant_mcp")
MAIN_ROOT = REPO / "quant/main"
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

import research_v260_fixed10_hold_optimization_20260822 as round1
import research_v260_fixed10_one_shot_2026_validation_runner_20260822 as runner


REPORTS = REPO / "quant/data_file/reports"
PROTOCOL_PATH = (
    REPORTS
    / "strategy_agent_v260_fixed10_one_shot_2026_validation_protocol_20260822/"
    "one_shot_validation_protocol.json"
)
VALIDATION_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_one_shot_2026_validation_20260822"
)
RESULT_PATH = VALIDATION_ROOT / "result.json"
FAILURE_PATH = VALIDATION_ROOT / "failure.json"
OUTPUT_ROOT = (
    REPORTS / "strategy_agent_v260_fixed10_one_shot_2026_result_audit_20260822"
)


def terminal_binding_checks(
    terminal: dict,
    protocol: dict,
    protocol_path: Path,
    start_marker_path: Path,
    start_claim_path: Path,
) -> dict[str, bool]:
    if (
        not protocol_path.is_file()
        or not start_marker_path.is_file()
        or not start_claim_path.is_file()
    ):
        return {
            "protocol_file_present": protocol_path.is_file(),
            "start_marker_present": start_marker_path.is_file(),
            "exclusive_start_claim_present": start_claim_path.is_file(),
            "terminal_protocol_sha256_exact": False,
            "terminal_start_marker_sha256_exact": False,
            "marker_protocol_sha256_exact": False,
            "marker_protocol_id_exact": False,
            "marker_validation_boundary_exact": False,
            "marker_exclusive_claim_sha256_exact": False,
        }
    marker = json.loads(start_marker_path.read_text(encoding="utf-8"))
    protocol_sha256 = runner.sha256(protocol_path)
    marker_sha256 = runner.sha256(start_marker_path)
    claim_sha256 = runner.sha256(start_claim_path)
    return {
        "protocol_file_present": True,
        "start_marker_present": True,
        "exclusive_start_claim_present": True,
        "terminal_protocol_sha256_exact": terminal.get("protocol_sha256")
        == protocol_sha256,
        "terminal_start_marker_sha256_exact": terminal.get(
            "start_marker_sha256"
        )
        == marker_sha256,
        "marker_protocol_sha256_exact": marker.get("protocol_sha256")
        == protocol_sha256,
        "marker_protocol_id_exact": marker.get("protocol_id")
        == protocol["protocol_id"],
        "marker_validation_boundary_exact": marker.get("validation_boundary")
        == protocol["validation_boundary"],
        "marker_exclusive_claim_sha256_exact": marker.get(
            "exclusive_claim_sha256"
        )
        == claim_sha256,
    }


def classify_business_result(metrics: dict, gates: dict | None) -> str:
    if gates is None or not all(gates.values()):
        return "reject"
    candidate = metrics["fixed10_current_candidate_without_event_overlay"]
    production = metrics["production_strategy_reference"]
    if candidate["cumulative_return"] <= production["cumulative_return"]:
        return "reject"
    if (
        candidate["sharpe"] >= production["sharpe"]
        and candidate["max_drawdown"] <= production["max_drawdown"]
    ):
        return "risk_adjusted_upgrade"
    return "return_upgrade_with_risk_tradeoff"


def metrics_equivalent(actual: object, expected: object) -> bool:
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            metrics_equivalent(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12
        )
    return actual == expected


def recompute_output_metrics(result: dict, protocol: dict) -> dict:
    validation_start = str(protocol["validation_boundary"]["start"])
    validation_end = str(protocol["validation_boundary"]["end"])
    event_start = str(
        protocol["event_overlay_runtime"]["diagnostic_coverage"][
            "event_metric_start"
        ]
    )
    sources = {
        "production_strategy_reference": (
            "production", validation_start, result["metrics"]
        ),
        "fixed10_current_candidate_without_event_overlay": (
            "fixed10", validation_start, result["metrics"]
        ),
        "fixed10_current_candidate_stress_0_65pct": (
            "fixed10_stress_0_65pct", validation_start, result["metrics"]
        ),
        "fixed10_current_candidate_with_frozen_event_overlay": (
            "fixed10_event_overlay", validation_start, result["metrics"]
        ),
        "fixed10_without_event_tiebreak": (
            "fixed10", event_start, result["event_overlay_coverage_metrics"]
        ),
        "fixed10_with_severe_event_tiebreak": (
            "fixed10_event_overlay",
            event_start,
            result["event_overlay_coverage_metrics"],
        ),
    }
    prefixes = sorted({value[0] for value in sources.values()})
    frames = {}
    artifact_boundary_checks = {}
    for prefix in prefixes:
        daily = pd.read_csv(
            VALIDATION_ROOT / f"{prefix}_daily.csv", dtype={"date": str}
        )
        actions = pd.read_csv(
            VALIDATION_ROOT / f"{prefix}_actions.csv",
            dtype={"buy_date": str, "action": str},
        )
        dates = daily["date"].astype(str)
        action_dates = actions["buy_date"].astype(str) if len(actions) else None
        artifact_boundary_checks[prefix] = bool(
            len(dates)
            and dates.min() == validation_start
            and dates.max() == validation_end
            and dates.between(validation_start, validation_end).all()
            and (
                action_dates is None
                or action_dates.between(validation_start, validation_end).all()
            )
        )
        frames[prefix] = (daily, actions)
    recomputed, checks = {}, {}
    for metric_name, (prefix, start, expected_group) in sources.items():
        daily, actions = frames[prefix]
        value = round1.evaluate_run(daily, actions, start, validation_end)
        recomputed[metric_name] = value
        checks[metric_name] = metrics_equivalent(
            value, expected_group[metric_name]
        )
    return {
        "metrics": recomputed,
        "checks": checks,
        "artifact_boundary_checks": artifact_boundary_checks,
        "all_recomputed_metrics_match": (
            all(checks.values()) and all(artifact_boundary_checks.values())
        ),
    }


def audit_result(result: dict, protocol: dict) -> dict:
    required_metric_arms = {
        "production_strategy_reference",
        "fixed10_current_candidate_without_event_overlay",
        "fixed10_current_candidate_stress_0_65pct",
        "fixed10_current_candidate_with_frozen_event_overlay",
    }
    required_event_arms = {
        "fixed10_without_event_tiebreak",
        "fixed10_with_severe_event_tiebreak",
    }
    metrics = result.get("metrics", {})
    event_metrics = result.get("event_overlay_coverage_metrics", {})
    deterministic = result.get("deterministic_replay", {})
    expected_gates = None
    if required_metric_arms == set(metrics):
        expected_gates = runner.evaluate_base_candidate_gates(
            metrics["fixed10_current_candidate_without_event_overlay"],
            metrics["production_strategy_reference"],
            metrics["fixed10_current_candidate_stress_0_65pct"],
            deterministic,
            protocol["base_candidate_decision"],
        )
    expected_status = (
        "passed_research_candidate"
        if expected_gates is not None and all(expected_gates.values())
        else "rejected_on_2026"
    )
    expected_boundary = [
        protocol["validation_boundary"]["start"],
        protocol["validation_boundary"]["end"],
    ]
    validation_start, validation_end = expected_boundary
    event_metric_start = protocol["event_overlay_runtime"][
        "diagnostic_coverage"
    ]["event_metric_start"]
    event_audit = result.get("event_audit", {})
    event_delta = result.get("event_overlay_diagnostic_delta_vs_base", {})
    business_classification = (
        classify_business_result(metrics, expected_gates)
        if expected_gates is not None
        else "reject"
    )
    checks = {
        "protocol_id_exact": result.get("protocol_id") == protocol["protocol_id"],
        "validation_boundary_exact": result.get("validation_boundary")
        == expected_boundary,
        "metric_arms_exact": set(metrics) == required_metric_arms,
        "event_metric_arms_exact": set(event_metrics) == required_event_arms,
        "primary_metric_boundaries_exact": (
            required_metric_arms == set(metrics)
            and all(
                str(value.get("start")) == validation_start
                and str(value.get("end")) == validation_end
                for value in metrics.values()
            )
        ),
        "event_metric_boundaries_exact": (
            required_event_arms == set(event_metrics)
            and all(
                str(value.get("start")) == str(event_metric_start)
                and str(value.get("end")) == validation_end
                for value in event_metrics.values()
            )
        ),
        "validation_access_endpoint_exact": (
            str(result.get("validation_access", {}).get("logical_max_date"))
            == validation_end
            and str(
                result.get("validation_access", {}).get(
                    "business_date_predicate_max"
                )
            )
            == validation_end
        ),
        "base_gates_recomputed_exact": expected_gates is not None
        and result.get("base_candidate_gates") == expected_gates,
        "terminal_status_matches_base_gates": result.get("status")
        == expected_status,
        "event_overlay_is_diagnostic_only": result.get("event_overlay_status")
        == "diagnostic_only_no_selection_decision",
        "event_delta_does_not_replace_base_metrics": (
            isinstance(event_delta, dict)
            and "cumulative_return" in event_delta
            and metrics.get("fixed10_current_candidate_without_event_overlay")
            is not None
        ),
        "event_coverage_exact": event_audit.get("diagnostic_coverage")
        == protocol["event_overlay_runtime"]["diagnostic_coverage"],
        "precoverage_not_treated_as_zero": event_audit.get(
            "precoverage_treated_as_no_event"
        )
        is False,
        "deterministic_keys_exact": set(deterministic)
        == {"base_daily", "base_actions", "overlay_daily", "overlay_actions"},
        "validation_opened_once_result": result.get("validation_2026_opened")
        is True,
        "production_unchanged": result.get("production_modified") is False,
        "trading_not_triggered": result.get("trading_triggered") is False,
    }
    return {
        "status": (
            "readonly_result_audit_passed"
            if all(checks.values())
            else "readonly_result_audit_failed"
        ),
        "checks": checks,
        "candidate_terminal_status": result.get("status"),
        "candidate_passed": result.get("status") == "passed_research_candidate",
        "candidate_business_classification": business_classification,
        "candidate_role": {
            "reject": "rejected",
            "return_upgrade_with_risk_tradeoff": (
                "return_superior_candidate_with_disclosed_risk_tradeoff"
            ),
            "risk_adjusted_upgrade": "production_replacement_candidate",
        }[business_classification],
        "base_candidate_gates": result.get("base_candidate_gates"),
        "event_overlay_status": result.get("event_overlay_status"),
        "validation_2026_opened": True,
        "production_modified": False,
    }


def audit_failure(failure: dict, protocol: dict) -> dict:
    checks = {
        "failure_status_exact": failure.get("status")
        == "failed_after_one_shot_validation_started",
        "protocol_id_exact": failure.get("protocol_id") == protocol["protocol_id"],
        "protocol_sha256_exact": failure.get("protocol_sha256")
        == runner.sha256(PROTOCOL_PATH),
        "validation_boundary_exact": failure.get("validation_boundary")
        == protocol["validation_boundary"],
        "validation_opened": failure.get("validation_2026_opened") is True,
        "retry_prohibited": failure.get("retry_allowed") is False,
        "production_unchanged": failure.get("production_modified") is False,
        "trading_not_triggered": failure.get("trading_triggered") is False,
        "exception_recorded": bool(failure.get("exception_type"))
        and bool(failure.get("exception_message"))
        and bool(failure.get("traceback")),
    }
    return {
        "status": (
            "readonly_failure_audit_passed"
            if all(checks.values())
            else "readonly_failure_audit_failed"
        ),
        "checks": checks,
        "candidate_terminal_status": "validation_failed_no_retry",
        "candidate_passed": False,
        "validation_2026_opened": True,
        "production_modified": False,
    }


def output_artifact_hashes() -> dict:
    paths = {
        name: VALIDATION_ROOT / name
        for name in runner.OUTPUT_ARTIFACT_NAMES
    }
    return {
        name: {
            "path": str(path).replace("\\", "/"),
            "exists": path.is_file(),
            "sha256": runner.sha256(path) if path.is_file() else None,
        }
        for name, path in paths.items()
    }


def output_artifact_digests_match(terminal: dict, artifacts: dict) -> bool:
    declared = terminal.get("output_artifact_sha256", {})
    return set(declared) == set(runner.OUTPUT_ARTIFACT_NAMES) and all(
        artifacts[name]["exists"]
        and artifacts[name]["sha256"] == declared[name]
        for name in runner.OUTPUT_ARTIFACT_NAMES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=RESULT_PATH)
    parser.add_argument("--failure", type=Path, default=FAILURE_PATH)
    args = parser.parse_args()
    protocol = runner.load_and_validate_protocol()
    result_exists = args.result.is_file()
    failure_exists = args.failure.is_file()
    if result_exists == failure_exists:
        raise RuntimeError("exactly one terminal validation artifact must exist")
    if result_exists:
        terminal = json.loads(args.result.read_text(encoding="utf-8"))
        audit = audit_result(terminal, protocol)
        audit["checks"].update(
            terminal_binding_checks(
                terminal,
                protocol,
                PROTOCOL_PATH,
                runner.START_MARKER,
                runner.START_CLAIM,
            )
        )
        if not all(audit["checks"].values()):
            audit["status"] = "readonly_result_audit_failed"
        audit["terminal_artifact"] = str(args.result).replace("\\", "/")
        audit["terminal_artifact_sha256"] = runner.sha256(args.result)
        audit["output_artifacts"] = output_artifact_hashes()
        audit["all_output_artifacts_present"] = all(
            value["exists"] for value in audit["output_artifacts"].values()
        )
        audit["checks"]["output_artifact_digests_exact"] = (
            output_artifact_digests_match(terminal, audit["output_artifacts"])
        )
        if audit["all_output_artifacts_present"]:
            recomputed = recompute_output_metrics(terminal, protocol)
            audit["independent_metric_recomputation"] = recomputed
        else:
            recomputed = {"all_recomputed_metrics_match": False}
        if (
            not all(audit["checks"].values())
            or
            not audit["all_output_artifacts_present"]
            or not recomputed["all_recomputed_metrics_match"]
        ):
            audit["status"] = "readonly_result_audit_failed"
    else:
        terminal = json.loads(args.failure.read_text(encoding="utf-8"))
        audit = audit_failure(terminal, protocol)
        audit["checks"].update(
            terminal_binding_checks(
                terminal,
                protocol,
                PROTOCOL_PATH,
                runner.START_MARKER,
                runner.START_CLAIM,
            )
        )
        if not all(audit["checks"].values()):
            audit["status"] = "readonly_failure_audit_failed"
        audit["terminal_artifact"] = str(args.failure).replace("\\", "/")
        audit["terminal_artifact_sha256"] = runner.sha256(args.failure)
    audit["protocol_path"] = str(PROTOCOL_PATH).replace("\\", "/")
    audit["protocol_sha256"] = runner.sha256(PROTOCOL_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    round1.atomic_json(OUTPUT_ROOT / "readonly_result_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()

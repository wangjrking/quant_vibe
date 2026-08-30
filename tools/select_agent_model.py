"""Route a prompt to the project-default GPT-5.6 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = ["terra"]
THINKING_ORDER = ["low", "medium", "high", "xhigh", "max", "ultra"]
DEFAULT_POLICY = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "agent_model_budget_routing_policy.json"
)


def _ratio(remaining: int | None, total: int | None) -> float | None:
    if remaining is None or total is None:
        return None
    if total <= 0 or remaining < 0:
        raise ValueError("budgets must be non-negative and totals must be positive")
    return min(1.0, remaining / total)


def _effective_ratio(*values: float | None) -> float | None:
    known = [value for value in values if value is not None]
    return min(known) if known else None


def _resource_model(policy: dict[str, Any], ratio: float) -> str | None:
    bands = sorted(
        policy["resource_bands"],
        key=lambda item: item["minimum_effective_ratio"],
        reverse=True,
    )
    for band in bands:
        if ratio >= band["minimum_effective_ratio"]:
            return band["maximum_model"]
    raise ValueError("resource bands do not cover ratio 0")


def _higher(left: str, right: str, order: list[str]) -> str:
    return left if order.index(left) >= order.index(right) else right


def _lower(left: str, right: str, order: list[str]) -> str:
    return left if order.index(left) <= order.index(right) else right


def _budget_payload(
    remaining_tokens: int | None,
    token_budget: int | None,
    remaining_seconds: int | None,
    time_budget_seconds: int | None,
) -> dict[str, Any]:
    token_ratio = _ratio(remaining_tokens, token_budget)
    time_ratio = _ratio(remaining_seconds, time_budget_seconds)
    return {
        "remaining_tokens": remaining_tokens,
        "token_budget": token_budget,
        "remaining_seconds": remaining_seconds,
        "time_budget_seconds": time_budget_seconds,
        "token_ratio": token_ratio,
        "time_ratio": time_ratio,
        "effective_ratio": _effective_ratio(token_ratio, time_ratio),
        "unknown_dimensions": [
            name
            for name, value in (("tokens", token_ratio), ("time", time_ratio))
            if value is None
        ],
    }


def _validate_terra_default_policy(policy: dict[str, Any]) -> None:
    """Reject policy drift away from the owner-directed Terra default."""
    models = policy.get("models", {})
    terra = models.get("terra")
    if set(models) != {"terra"} or not terra or terra.get("model") != "gpt-5.6-terra":
        raise ValueError("terra_default_policy_requires_gpt_5.6_terra")
    fixed_dispatch = policy.get("fixed_dispatch") or {}
    if fixed_dispatch.get("enabled") and fixed_dispatch.get("model_key") != "terra":
        raise ValueError("fixed_dispatch_must_use_terra")
    for profile_group in ("complexity_profiles", "speed_profiles", "risk_profiles"):
        for profile in policy.get(profile_group, {}).values():
            for key in ("default_model", "low_risk_model_preference", "minimum_model"):
                if key in profile and profile[key] not in {"terra", None}:
                    raise ValueError("terra_default_policy_contains_non_terra_route")
    for band in policy.get("resource_bands", []):
        if band.get("maximum_model") not in {"terra", None}:
            raise ValueError("terra_default_policy_contains_non_terra_resource_cap")


def _blocked(
    policy: dict[str, Any],
    complexity: int,
    speed: str,
    risk: str,
    budget: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "policy_id": policy["policy_id"],
        "decision": "block",
        "status": "blocked_budget_insufficient",
        "complexity": complexity,
        "speed": speed,
        "risk": risk,
        "selected_model": None,
        "thinking": None,
        "reason": reason,
        "budget": budget,
    }


def select_model(
    policy: dict[str, Any],
    complexity: int,
    speed: str,
    risk: str,
    remaining_tokens: int | None = None,
    token_budget: int | None = None,
    remaining_seconds: int | None = None,
    time_budget_seconds: int | None = None,
) -> dict[str, Any]:
    _validate_terra_default_policy(policy)
    profile = policy["complexity_profiles"].get(str(complexity))
    if profile is None:
        raise ValueError("complexity must be between 1 and 5")
    if speed not in policy["speed_profiles"]:
        raise ValueError(f"unknown speed: {speed}")
    if risk not in policy["risk_profiles"]:
        raise ValueError(f"unknown risk: {risk}")

    speed_profile = policy["speed_profiles"][speed]
    risk_profile = policy["risk_profiles"][risk]
    budget = _budget_payload(
        remaining_tokens,
        token_budget,
        remaining_seconds,
        time_budget_seconds,
    )

    fixed_dispatch = policy.get("fixed_dispatch")
    if fixed_dispatch and fixed_dispatch.get("enabled"):
        model_key = fixed_dispatch["model_key"]
        thinking = fixed_dispatch["thinking"]
        if model_key not in policy["models"]:
            raise ValueError("fixed dispatch model is not configured")
        if thinking not in policy["models"][model_key]["allowed_thinking"]:
            raise ValueError("fixed dispatch thinking is not allowed for the configured model")
        return {
            "policy_id": policy["policy_id"],
            "decision": "dispatch",
            "status": "selected_fixed_dispatch",
            "complexity": complexity,
            "speed": speed,
            "risk": risk,
            "selected_model": policy["models"][model_key]["model"],
            "thinking": thinking,
            "reason": fixed_dispatch["reason"],
            "budget": budget,
        }

    below_absolute_floor = (
        remaining_tokens is not None
        and remaining_tokens < profile["minimum_remaining_tokens"]
    ) or (
        remaining_seconds is not None
        and remaining_seconds < profile["minimum_remaining_seconds"]
    )
    if below_absolute_floor:
        return _blocked(
            policy,
            complexity,
            speed,
            risk,
            budget,
            "remaining token or time budget is below the complexity safety floor",
        )

    model_key = profile["default_model"]
    model_key = _higher(model_key, risk_profile["minimum_model"], MODEL_ORDER)
    thinking = profile["default_thinking"]
    if risk_profile.get("minimum_thinking"):
        thinking = _higher(
            thinking,
            risk_profile["minimum_thinking"],
            THINKING_ORDER,
        )

    speed_preference = speed_profile["low_risk_model_preference"]
    if speed_preference and risk in {"low", "normal"} and complexity <= 3:
        model_key = _lower(model_key, speed_preference, MODEL_ORDER)
        thinking = _lower(
            thinking,
            speed_profile["thinking_cap"],
            THINKING_ORDER,
        )

    effective_ratio = budget["effective_ratio"]
    if effective_ratio is not None:
        resource_cap = _resource_model(policy, effective_ratio)
        if resource_cap is None:
            return _blocked(
                policy,
                complexity,
                speed,
                risk,
                budget,
                "less than 10% of the token or time budget remains",
            )
        if MODEL_ORDER.index(model_key) > MODEL_ORDER.index(resource_cap):
            minimum_model = risk_profile["minimum_model"]
            if (
                risk_profile.get("never_downgrade")
                or MODEL_ORDER.index(resource_cap) < MODEL_ORDER.index(minimum_model)
            ):
                return _blocked(
                    policy,
                    complexity,
                    speed,
                    risk,
                    budget,
                    "resource cap is below the risk-required GPT-5.6 model",
                )
            model_key = resource_cap

    allowed_thinking = policy["models"][model_key]["allowed_thinking"]
    while thinking not in allowed_thinking:
        thinking = THINKING_ORDER[THINKING_ORDER.index(thinking) - 1]

    return {
        "policy_id": policy["policy_id"],
        "decision": "dispatch",
        "status": "selected",
        "complexity": complexity,
        "speed": speed,
        "risk": risk,
        "selected_model": policy["models"][model_key]["model"],
        "thinking": thinking,
        "reason": (
            "selected from prompt complexity, required speed, risk floor, "
            "and the minimum known remaining resource ratio"
        ),
        "budget": budget,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--complexity", required=True, type=int, choices=range(1, 6))
    parser.add_argument(
        "--speed",
        required=True,
        choices=["immediate", "fast", "normal", "deep"],
    )
    parser.add_argument(
        "--risk",
        required=True,
        choices=["low", "normal", "high", "critical"],
    )
    parser.add_argument("--remaining-tokens", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--remaining-seconds", type=int)
    parser.add_argument("--time-budget-seconds", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    result = select_model(
        policy=policy,
        complexity=args.complexity,
        speed=args.speed,
        risk=args.risk,
        remaining_tokens=args.remaining_tokens,
        token_budget=args.token_budget,
        remaining_seconds=args.remaining_seconds,
        time_budget_seconds=args.time_budget_seconds,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["decision"] == "dispatch" else 2


if __name__ == "__main__":
    raise SystemExit(main())

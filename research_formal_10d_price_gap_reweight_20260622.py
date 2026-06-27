from __future__ import annotations

from pathlib import Path

import research_formal_10d_bucket_risk_reweight_20260622 as base


ROOT = Path(r"D:\work\quant\quant_mcp")
base.REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_price_gap_reweight_turn6_20260622"
)
base.SOURCES["turn6"] = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal_10d_bucket_risk_reweight_v2_20260622"
    / "signals"
    / "turn6_down85.csv"
)


def v(name: str, rules: list[dict], *, cap: float = 0.42, source: str = "turn6", floor: float | None = None, env: dict[str, str] | None = None) -> dict:
    return base._variant(
        name,
        rules,
        source=source,
        cap=cap,
        floor_day_sum=floor,
        extra_env=env,
    )


base.VARIANTS = [
    v("repro", []),
    v(
        "close6_down88_close58_boost106",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.88},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.06},
        ],
    ),
    v(
        "close8_down82_close58_boost108",
        [
            {"field": "close", "op": "ge", "threshold": 8.0, "scale": 0.82},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.08},
        ],
    ),
    v(
        "gap063_down78_gap006_boost108",
        [
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.78},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.08},
        ],
    ),
    v(
        "gap05_down85_gap006_boost108",
        [
            {"field": "pred_gap", "op": "ge", "threshold": 0.05, "scale": 0.85},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.08},
        ],
    ),
    v(
        "p10high_down80_p10mid_boost106",
        [
            {"field": "pred_10d", "op": "ge", "threshold": 0.036, "scale": 1.06},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.75},
        ],
    ),
    v(
        "p10high_down75_p10low_boost105",
        [
            {"field": "pred_10d", "op": "lt", "threshold": 0.02, "scale": 1.05},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.75},
        ],
    ),
    v(
        "p5mid_boost108_p5high_down88",
        [
            {"field": "pred_5d", "op": "ge", "threshold": 0.007, "scale": 1.08},
            {"field": "pred_5d", "op": "ge", "threshold": 0.028, "scale": 0.82},
        ],
    ),
    v(
        "close_gap_combo_soft",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.92},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.05},
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.84},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.05},
        ],
    ),
    v(
        "close_gap_p10_combo",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.92},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.05},
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.84},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.05},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.82},
        ],
    ),
    v(
        "close_gap_p10_p5_combo",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.92},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.05},
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.84},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.05},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.82},
            {"field": "pred_5d", "op": "ge", "threshold": 0.007, "scale": 1.05},
            {"field": "pred_5d", "op": "ge", "threshold": 0.028, "scale": 0.84},
        ],
    ),
    v(
        "close_gap_p10_p5_cap40",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.92},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.05},
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.84},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.05},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.82},
            {"field": "pred_5d", "op": "ge", "threshold": 0.007, "scale": 1.05},
            {"field": "pred_5d", "op": "ge", "threshold": 0.028, "scale": 0.84},
        ],
        cap=0.40,
        floor=0.95,
    ),
    v(
        "close_gap_p10_no_dd",
        [
            {"field": "close", "op": "ge", "threshold": 6.0, "scale": 0.92},
            {"field": "close", "op": "lt", "threshold": 5.8, "scale": 1.05},
            {"field": "pred_gap", "op": "ge", "threshold": 0.063, "scale": 0.84},
            {"field": "pred_gap", "op": "lt", "threshold": 0.006, "scale": 1.05},
            {"field": "pred_10d", "op": "ge", "threshold": 0.079, "scale": 0.82},
        ],
        env={"GM_EQUITY_DD_RISK_MODE": "0"},
    ),
]


if __name__ == "__main__":
    raise SystemExit(base.main())

"""Scan strict pre-2022 train-only feature stability for the 10D research path."""

from __future__ import annotations

import json
from pathlib import Path

from fast_feature_selection import score_features_fast_split


OUT = Path(__file__).resolve().parents[1] / "data_file" / "reports" / "model_agent_10d_train_only_feature_stability_20260830"


def main() -> int:
    if OUT.exists():
        raise RuntimeError("blocked_fail_closed_output_exists")
    OUT.mkdir(parents=True)
    rows, selected = score_features_fast_split(
        "../data_file/production_assets/duckdb/l3_feature_current.duckdb",
        label_path="../data_file/production_assets/duckdb/l3_label_current.duckdb",
        feature_table="prod_l3_production_factor_parts_20260625",
        label_table="prod_l3_prediction_label_parts_current",
        label="executable_10d_open_return",
        start="20100104",
        end="20211217",
        top_n=80,
        min_abs_ic=0.005,
        max_missing_ratio=0.35,
        folds=8,
    )
    result = {
        "selection_window": ["20100104", "20211217"],
        "development_and_sealed_windows": {"2022_2024": "not_read", "2025": "not_read", "2026_plus": "not_read"},
        "label": "executable_10d_open_return",
        "selection_config": {"top_n": 80, "min_abs_ic": 0.005, "max_missing_ratio": 0.35, "contiguous_train_segments": 8},
        "rows": rows,
        "selected": selected,
        "production_unchanged": True,
        "allow_next_layer_continue": False,
    }
    (OUT / "selection_result.json").write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

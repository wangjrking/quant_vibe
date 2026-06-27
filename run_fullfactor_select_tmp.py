from __future__ import annotations

import json
from pathlib import Path

from fast_feature_selection import score_features_fast_parquet, write_score_csv


OUT = Path(
    r"D:\work\quant\quant_mcp\quant\data_file\reports\fullfactor_expanding2010_20260616\global_feature_select_20100101_20240524_no_gtja"
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, selected = score_features_fast_parquet(
        r"D:\work\quant\quant_mcp\quant\data_file\stock_factor_data.parquet",
        label="executable_10d_open_return",
        start="20100101",
        end="20240524",
        top_n=220,
        min_abs_ic=0.003,
        max_missing_ratio=0.45,
        folds=8,
        exclude_prefixes=("gtja_alpha",),
        chunk_size=30,
    )
    write_score_csv(rows, OUT / "feature_ic_scores_executable_10d_open_return_top220.csv")
    for top_n in (80, 120, 160, 220):
        payload = {
            "label": "executable_10d_open_return",
            "features": selected[:top_n],
            "selection_window": {"start": "20100101", "end": "20240524"},
            "excluded_prefixes": ["gtja_alpha"],
        }
        (OUT / f"selected_features_executable_10d_open_return_top{top_n}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print("selected_count", len(selected), flush=True)
    print("top20", selected[:20], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

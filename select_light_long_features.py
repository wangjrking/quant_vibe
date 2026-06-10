from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from feature_selection_module import FeatureSelectionConfig, score_features
from light_factor_module import build_light_factor_frame, read_raw_frame


DEFAULT_CANDIDATES = [
    "close",
    "open",
    "high",
    "low",
    "pre_close",
    "vol",
    "amount",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "circ_mv",
    "total_mv",
    "pb",
    "pe",
    "ps",
    "dv_ttm",
    "atr_qfq",
    "fd_amount",
    "open_times",
    "buy_sm_amount",
    "sell_sm_amount",
    "buy_md_amount",
    "sell_md_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
    "net_mf_amount",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
    "winner_rate",
    "index_2000_close",
    "index_2000_open",
    "index_2000_high",
    "index_2000_low",
    "index_2000_amount",
    "close_rate",
    "open_rate",
    "high_rate",
    "low_rate",
    "high_open_rate",
    "high_close_rate",
    "low_open_rate",
    "low_close_rate",
    "diff_close_low",
    "diff_close_high",
    "diff_high_low",
    "industry_encode",
    "stock_encode",
]

DEFAULT_FORCE_INCLUDE = [
    "amount",
    "turnover_rate",
    "volume_ratio",
    "circ_mv",
    "total_mv",
    "atr_qfq",
]


def _parse_csv_list(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return list(fallback)
    return [item.strip() for item in value.split(",") if item.strip()]


def _write_score_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def prune_correlated_features(
    frame: pd.DataFrame,
    ranked_features: list[str],
    force_include: list[str],
    max_abs_corr: float,
    top_n: int,
) -> list[str]:
    selected: list[str] = []
    force_include = [feature for feature in force_include if feature in frame.columns]
    for feature in force_include:
        if feature not in selected:
            selected.append(feature)

    for feature in ranked_features:
        if feature in selected or feature not in frame.columns:
            continue
        keep = True
        values = pd.to_numeric(frame[feature], errors="coerce")
        for chosen in selected:
            corr = values.corr(pd.to_numeric(frame[chosen], errors="coerce"))
            if corr is not None and pd.notna(corr) and abs(float(corr)) >= max_abs_corr:
                keep = False
                break
        if keep:
            selected.append(feature)
        if len(selected) >= top_n:
            break
    return selected[:top_n]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select long-horizon light features with IC ranking and redundancy pruning.")
    parser.add_argument("--data-dir", default="../data_file")
    parser.add_argument("--train-start", default="20200101")
    parser.add_argument("--test-start", default="20240604")
    parser.add_argument("--end", default="20260605")
    parser.add_argument("--label", default="executable_10d_open_return")
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--min-abs-ic", type=float, default=0.003)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--max-abs-corr", type=float, default=0.92)
    parser.add_argument("--candidates")
    parser.add_argument("--force-include")
    parser.add_argument("--output", default="../data_file/selected_features_executable_10d_open_return_light_long_v2.json")
    parser.add_argument("--score-output", default="../data_file/feature_ic_scores_executable_10d_open_return_light_long_v2.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    candidate_features = _parse_csv_list(args.candidates, DEFAULT_CANDIDATES)
    force_include = _parse_csv_list(args.force_include, DEFAULT_FORCE_INCLUDE)
    needed_columns = list(dict.fromkeys(candidate_features + ["trade_date", "stock_code", "name", "industry", "st_type", "limit_times", "open", "high", "low", "close", "pre_close", "atr_qfq"]))
    raw = read_raw_frame(
        data_dir / "odb.db",
        start=args.train_start,
        end=args.end,
        needed_columns=needed_columns,
        stock_pool_path=None,
    )
    factor = build_light_factor_frame(raw, candidate_features, args.label)
    train_frame = factor[factor["trade_date"].astype(str) < args.test_start].copy()
    rows = score_features(
        train_frame,
        FeatureSelectionConfig(
            label=args.label,
            top_n=max(args.top_n * 3, args.top_n),
            min_abs_ic=args.min_abs_ic,
            max_missing_ratio=args.max_missing_ratio,
            start_date=args.train_start,
            end_date=None,
            candidate_features=candidate_features,
        ),
    )
    ranked = [row["feature"] for row in rows if row["abs_mean_ic"] >= args.min_abs_ic]
    selected = prune_correlated_features(
        train_frame,
        ranked_features=ranked,
        force_include=force_include,
        max_abs_corr=args.max_abs_corr,
        top_n=args.top_n,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"label": args.label, "features": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_score_csv(rows, Path(args.score_output))
    print(f"candidate_features={len(candidate_features)} ranked={len(ranked)} selected={len(selected)} output={output}")
    print("selected_list=" + ",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

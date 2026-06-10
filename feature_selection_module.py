"""IC-based feature selection for return-focused model training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from leakage_guard import find_leaky_features


@dataclass
class FeatureSelectionConfig:
    label: str = "10d_yield_rate"
    top_n: int = 160
    min_abs_ic: float = 0.005
    max_missing_ratio: float = 0.35
    start_date: str | None = None
    end_date: str | None = None
    candidate_features: list[str] | None = None


def _safe_spearman(feature: pd.Series, label: pd.Series) -> float | None:
    frame = pd.DataFrame({"feature": feature, "label": label}).dropna()
    if len(frame) < 3 or frame["feature"].nunique() < 2 or frame["label"].nunique() < 2:
        return None
    value = frame["feature"].rank().corr(frame["label"].rank())
    if value is None or math.isnan(value):
        return None
    return float(value)


def _date_filter(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if "trade_date" not in frame.columns:
        return frame
    dates = frame["trade_date"].astype(str)
    mask = pd.Series(True, index=frame.index)
    if start_date:
        mask &= dates >= start_date
    if end_date:
        mask &= dates <= end_date
    return frame.loc[mask]


def prepare_selection_label(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if label == "risk_adjusted_10d_yield_rate":
        base_return = pd.to_numeric(frame["10d_yield_rate"], errors="coerce")
        atr_ratio = (
            pd.to_numeric(frame["atr_qfq"], errors="coerce")
            / pd.to_numeric(frame["close"], errors="coerce")
        ).clip(lower=0.01, upper=0.20)
        frame[label] = base_return / atr_ratio
    elif label == "executable_10d_open_return":
        buy = pd.to_numeric(frame["post_open"], errors="coerce")
        sell = pd.to_numeric(frame["post12_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_5d_open_return":
        buy = pd.to_numeric(frame["post_open"], errors="coerce")
        sell = pd.to_numeric(frame["post6_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_3d_open_return":
        buy = pd.to_numeric(frame["post_open"], errors="coerce")
        sell = pd.to_numeric(frame["post4_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "executable_1d_open_return":
        buy = pd.to_numeric(frame["post_open"], errors="coerce")
        sell = pd.to_numeric(frame["post2_open"], errors="coerce")
        entry_cash = buy * (1.0 + 0.0003 + 0.001)
        exit_cash = sell * (1.0 - 0.0003 - 0.0005 - 0.001)
        frame[label] = exit_cash / entry_cash - 1.0
    elif label == "excess_10d_yield_rate":
        frame[label] = pd.to_numeric(frame["adjust_10d_yield_rate"], errors="coerce")
    return frame


def _numeric_candidates(frame: pd.DataFrame, config: FeatureSelectionConfig) -> list[str]:
    if config.candidate_features:
        candidates = [feature for feature in config.candidate_features if feature in frame.columns]
    else:
        excluded = {"stock_code", "trade_date", "name", "industry", "act_ent_type", config.label}
        candidates = [col for col in frame.columns if col not in excluded]
    leaky = set(find_leaky_features(candidates, label=config.label))
    result = []
    for feature in candidates:
        if feature in leaky:
            continue
        series = pd.to_numeric(frame[feature], errors="coerce")
        if series.notna().mean() < 1.0 - config.max_missing_ratio:
            continue
        result.append(feature)
    return result


def score_features(frame: pd.DataFrame, config: FeatureSelectionConfig) -> list[dict]:
    frame = prepare_selection_label(frame.copy(), config.label)
    frame = _date_filter(frame, config.start_date, config.end_date)
    if config.label not in frame.columns:
        raise ValueError(f"label column not found: {config.label}")
    label = pd.to_numeric(frame[config.label], errors="coerce")
    features = _numeric_candidates(frame, config)
    rows = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        daily_ics = []
        for _, group in pd.DataFrame({"trade_date": frame["trade_date"], "feature": values, "label": label}).groupby("trade_date"):
            ic = _safe_spearman(group["feature"], group["label"])
            if ic is not None:
                daily_ics.append(ic)
        if not daily_ics:
            continue
        mean_ic = sum(daily_ics) / len(daily_ics)
        std_ic = pd.Series(daily_ics).std()
        ic_ir = 0.0 if not std_ic or math.isnan(std_ic) else mean_ic / std_ic
        missing_ratio = float(values.isna().mean())
        rows.append(
            {
                "feature": feature,
                "mean_ic": mean_ic,
                "abs_mean_ic": abs(mean_ic),
                "ic_ir": ic_ir,
                "missing_ratio": missing_ratio,
                "ic_days": len(daily_ics),
            }
        )
    rows.sort(key=lambda row: (row["abs_mean_ic"], abs(row["ic_ir"]), -row["missing_ratio"]), reverse=True)
    return rows


def select_features(frame: pd.DataFrame, config: FeatureSelectionConfig) -> list[str]:
    scored = score_features(frame, config)
    selected = [
        row["feature"]
        for row in scored
        if row["abs_mean_ic"] >= config.min_abs_ic
    ][: config.top_n]
    return selected


def write_score_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Select training features by daily IC.")
    parser.add_argument("--data", default="../data_file/stock_factor_data.parquet")
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--top-n", type=int, default=160)
    parser.add_argument("--min-abs-ic", type=float, default=0.005)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", default="../data_file/selected_features_10d_yield_rate.json")
    parser.add_argument("--score-output", default="../data_file/feature_ic_scores_10d_yield_rate.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frame = pd.read_parquet(args.data)
    config = FeatureSelectionConfig(
        label=args.label,
        top_n=args.top_n,
        min_abs_ic=args.min_abs_ic,
        max_missing_ratio=args.max_missing_ratio,
        start_date=args.start,
        end_date=args.end,
    )
    rows = score_features(frame, config)
    selected = [row["feature"] for row in rows if row["abs_mean_ic"] >= config.min_abs_ic][: config.top_n]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"label": args.label, "features": selected}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_score_csv(rows, Path(args.score_output))
    print(f"selected_features={len(selected)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

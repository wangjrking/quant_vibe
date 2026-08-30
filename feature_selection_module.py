"""IC-based feature selection for return-focused model training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from adjustment_semantics import require_front_adjusted_column
from leakage_guard import find_leaky_features
from model_asset_route import (
    MODEL_FEATURE_MODE_LEGACY,
    MODEL_FEATURE_MODE_SPLIT,
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
)


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
    from ai_module import prepare_training_label

    if label in frame.columns:
        frame[label] = pd.to_numeric(frame[label], errors="coerce")
        return frame
    if label == "risk_adjusted_10d_yield_rate":
        base_return = pd.to_numeric(frame["10d_yield_rate"], errors="coerce")
        close_column = require_front_adjusted_column(
            frame.columns,
            "close",
            context="risk_adjusted_10d_yield_rate",
        )
        atr_ratio = (
            pd.to_numeric(frame["atr_qfq"], errors="coerce")
            / pd.to_numeric(frame[close_column], errors="coerce")
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
    elif label.startswith("executable_") and ("_top" in label):
        frame = prepare_training_label(frame, label)
    return frame


def _split_label_required_columns(label: str) -> list[str]:
    if label == "risk_adjusted_10d_yield_rate":
        return ["10d_yield_rate"]
    if label == "excess_10d_yield_rate":
        return ["adjust_10d_yield_rate"]
    if label.startswith("executable_") and ("_top" in label):
        if "_10d_" in label:
            return ["executable_10d_open_return"]
        if "_5d_" in label:
            return ["executable_5d_open_return"]
        if "_3d_" in label:
            return ["executable_3d_open_return"]
        if "_1d_" in label:
            return ["executable_1d_open_return"]
    return [label]


def _split_feature_required_columns(label: str) -> list[str]:
    if label == "risk_adjusted_10d_yield_rate":
        return ["atr_qfq", "close_qfq"]
    return []


def _quote_ident(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _resolve_duckdb_binding(
    data_path: str | Path | None,
    label_path: str | Path | None,
) -> tuple[Path, str, Path, str] | None:
    explicit_feature = Path(data_path) if data_path not in (None, "") else None
    explicit_label = Path(label_path) if label_path not in (None, "") else None
    feature_table = resolve_model_feature_duckdb_table(None)
    label_table = resolve_model_label_duckdb_table(None)
    if not feature_table or not label_table:
        return None
    feature_db_path = (
        explicit_feature
        if explicit_feature and explicit_feature.suffix.lower() == ".duckdb"
        else resolve_model_feature_duckdb_path(None, require_exists=True)
    )
    label_db_path = (
        explicit_label
        if explicit_label and explicit_label.suffix.lower() == ".duckdb"
        else resolve_model_label_duckdb_path(None, require_exists=True)
    )
    return feature_db_path, feature_table, label_db_path, label_table


def _duckdb_columns(db_path: Path, table: str) -> set[str]:
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as conn:
        return {str(row[0]) for row in conn.execute(f"DESCRIBE {_quote_ident(table)}").fetchall()}


def _read_duckdb_frame(
    db_path: Path,
    table: str,
    columns: list[str],
) -> pd.DataFrame:
    import duckdb

    selected = ", ".join(_quote_ident(column) for column in columns)
    sql = f"SELECT {selected} FROM {_quote_ident(table)}"
    with duckdb.connect(str(db_path), read_only=True) as conn:
        return conn.execute(sql).fetchdf()


def load_selection_frame(
    data_path: str | Path,
    *,
    label_path: str | Path | None,
    label: str,
    feature_source: str,
) -> pd.DataFrame:
    if feature_source == MODEL_FEATURE_MODE_LEGACY:
        return pd.read_parquet(data_path)
    duckdb_binding = _resolve_duckdb_binding(data_path, label_path)
    data_columns = ["trade_date", "stock_code", "name", "industry", "act_ent_type", *_split_feature_required_columns(label)]
    label_columns = [
        "trade_date",
        "stock_code",
        "5d_yield_rate",
        "open6_yield_rate",
        "10d_yield_rate",
        *_split_label_required_columns(label),
    ]
    if duckdb_binding is not None:
        feature_db_path, feature_table, label_db_path, label_table = duckdb_binding
        feature_columns = [
            column for column in list(dict.fromkeys(data_columns))
            if column in _duckdb_columns(feature_db_path, feature_table)
        ]
        resolved_label_columns = [
            column for column in list(dict.fromkeys(label_columns))
            if column in _duckdb_columns(label_db_path, label_table)
        ]
        factors = _read_duckdb_frame(feature_db_path, feature_table, feature_columns)
        labels = _read_duckdb_frame(label_db_path, label_table, resolved_label_columns)
    else:
        factors = pd.read_parquet(data_path, columns=list(dict.fromkeys(data_columns)))
        labels = pd.read_parquet(label_path, columns=list(dict.fromkeys(label_columns)))
    return factors.merge(labels, on=["trade_date", "stock_code"], how="inner")


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
    parser.add_argument("--data", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument(
        "--feature-source",
        default=MODEL_FEATURE_MODE_SPLIT,
        choices=[MODEL_FEATURE_MODE_SPLIT, MODEL_FEATURE_MODE_LEGACY],
    )
    parser.add_argument("--label", default="10d_yield_rate")
    parser.add_argument("--top-n", type=int, default=160)
    parser.add_argument("--min-abs-ic", type=float, default=0.005)
    parser.add_argument("--max-missing-ratio", type=float, default=0.35)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output", default="data_file/selected_features_10d_yield_rate.json")
    parser.add_argument("--score-output", default="data_file/feature_ic_scores_10d_yield_rate.csv")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    frame = load_selection_frame(
        args.data,
        label_path=args.labels,
        label=args.label,
        feature_source=args.feature_source,
    )
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

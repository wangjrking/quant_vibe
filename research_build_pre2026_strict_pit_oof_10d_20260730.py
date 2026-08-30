from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_pre2026_strict_pit_oof_10d_20260730"
EXPERIMENT_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "l4_predictions"
EXPERIMENT_DB = EXPERIMENT_DIR / "l4_pre2026_strict_pit_oof_10d_20260730.duckdb"
EXPERIMENT_TABLE = "stock_predict_data_model_agent_pre2026_strict_pit_oof_10d_20260730_research"
FEATURE_SRC_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_SRC_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_SRC_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_SRC_TABLE = "prod_l3_prediction_label_parts_current"
STRICT_FEATURE_DB = REPORT_DIR / "inputs" / "l3_feature_pre2026_strict_pit.duckdb"
STRICT_FEATURE_TABLE = "prod_l3_feature_pre2026_strict_pit"
STRICT_LABEL_DB = REPORT_DIR / "inputs" / "l3_label_pre2026_strict_pit.duckdb"
STRICT_LABEL_TABLE = "prod_l3_label_pre2026_strict_pit"
FOLD_PREDICTION_DIR = REPORT_DIR / "fold_predictions"
FOLD_SELECTION_DIR = REPORT_DIR / "feature_selection"
FOLD_SUMMARY_CSV = REPORT_DIR / "fold_training_summary.csv"
REPORT_JSON = REPORT_DIR / "pre2026_strict_pit_oof_10d_report.json"
REPORT_MD = REPORT_DIR / "pre2026_strict_pit_oof_10d_report.md"
MANIFEST_JSON = REPORT_DIR / "research_candidate_manifest.json"
HASH_INVENTORY_JSON = REPORT_DIR / "hash_inventory.json"
FOLD_DEFINITIONS_JSON = REPORT_DIR / "fold_definitions.json"
HANDOFF_JSON = REPORT_DIR / "audit_handoff.json"
HANDOFF_MD = REPORT_DIR / "audit_handoff.md"
PRECHECK_JSON = REPORT_DIR / "preflight_status.json"
CN_TZ = timezone(timedelta(hours=8))

TRAIN_HISTORY_START = "20100104"
SCORE_START = "20220606"
SCORE_END = "20251231"
TRAIN_YEARS = 4
TEST_MONTHS = 3
STEP_MONTHS = 3
EMBARGO_DAYS = 10
LABEL = "executable_10d_open_return"
FEATURE_TOP_N = 40
MIN_ABS_IC = 0.005
MAX_MISSING_RATIO = 0.35
FEATURE_IC_FOLDS = 8
MODEL_TYPE = "reg"

if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from model_asset_route import (  # noqa: E402
    ENV_MODEL_FEATURE_DUCKDB,
    ENV_MODEL_FEATURE_DUCKDB_TABLE,
    ENV_MODEL_LABEL_DUCKDB,
    ENV_MODEL_LABEL_DUCKDB_TABLE,
    MODEL_FEATURE_MODE_SPLIT,
    write_prediction_manifest,
)
from rolling_train_module import (  # noqa: E402
    FoldFeatureSelectionConfig,
    build_fold_feature_selection_fn,
    build_rolling_windows,
    train_one_fold_with_ai,
)


def now_iso() -> str:
    return datetime.now(CN_TZ).replace(microsecond=0).isoformat()


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def quote_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_progress(message: str) -> None:
    print(f"[{now_iso()}] {message}", flush=True)


@contextlib.contextmanager
def temporary_env(bindings: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in bindings}
    try:
        for key, value in bindings.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def duckdb_describe(db_path: Path, table: str) -> list[dict[str, Any]]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(f"DESCRIBE {quote_ident(table)}").fetchall()
    return [{"column": str(row[0]), "type": str(row[1])} for row in rows]


def read_trade_dates(db_path: Path, table: str, *, max_date: str) -> list[str]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT trade_date
            FROM {quote_ident(table)}
            WHERE trade_date <= ?
              AND stock_code NOT LIKE '%.BJ'
            ORDER BY trade_date
            """,
            [max_date],
        ).fetchall()
    return [str(row[0]) for row in rows]


def compute_mature_cutoff(trade_dates: list[str], horizon_trading_days: int) -> str:
    required_idx = len(trade_dates) - (horizon_trading_days + 1)
    if required_idx < 0:
        raise RuntimeError("insufficient trade dates to compute mature cutoff")
    return str(trade_dates[required_idx])


def materialize_strict_input_assets() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STRICT_FEATURE_DB.parent.mkdir(parents=True, exist_ok=True)
    log_progress("preflight_start_read_trade_dates")
    trade_dates = read_trade_dates(FEATURE_SRC_DB, FEATURE_SRC_TABLE, max_date=SCORE_END)
    if not trade_dates:
        raise RuntimeError("no trade dates found in active feature source before 20251231")
    mature_label_max = compute_mature_cutoff(trade_dates, horizon_trading_days=12)

    log_progress("materialize_strict_feature_db")
    with duckdb.connect(str(STRICT_FEATURE_DB)) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(STRICT_FEATURE_TABLE)}")
        conn.execute(
            f"""
            ATTACH {quote_literal(FEATURE_SRC_DB)} AS feature_src (READ_ONLY);
            CREATE TABLE {quote_ident(STRICT_FEATURE_TABLE)} AS
            SELECT *
            FROM feature_src.{quote_ident(FEATURE_SRC_TABLE)}
            WHERE trade_date >= '{TRAIN_HISTORY_START}'
              AND trade_date <= '{SCORE_END}'
              AND stock_code NOT LIKE '%.BJ';
            DETACH feature_src;
            """
        )

    log_progress("materialize_strict_label_db")
    with duckdb.connect(str(STRICT_LABEL_DB)) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {quote_ident(STRICT_LABEL_TABLE)}")
        conn.execute(
            f"""
            ATTACH {quote_literal(LABEL_SRC_DB)} AS label_src (READ_ONLY);
            CREATE TABLE {quote_ident(STRICT_LABEL_TABLE)} AS
            SELECT *
            FROM label_src.{quote_ident(LABEL_SRC_TABLE)}
            WHERE trade_date >= '{TRAIN_HISTORY_START}'
              AND trade_date <= '{SCORE_END}'
              AND stock_code NOT LIKE '%.BJ';
            DETACH label_src;
            """
        )
        label_columns_meta = conn.execute(f"DESCRIBE {quote_ident(STRICT_LABEL_TABLE)}").fetchall()
        label_columns = [
            str(row[0])
            for row in label_columns_meta
            if str(row[0]) not in {"trade_date", "stock_code"}
        ]
        if label_columns:
            set_clause = ", ".join(f"{quote_ident(column)} = NULL" for column in label_columns)
            conn.execute(
                f"""
                UPDATE {quote_ident(STRICT_LABEL_TABLE)}
                SET {set_clause}
                WHERE trade_date > ?
                """,
                [mature_label_max],
            )

    log_progress("probe_strict_input_quality")
    feature_quality = table_quality(STRICT_FEATURE_DB, STRICT_FEATURE_TABLE, value_column=None)
    label_quality = table_quality(STRICT_LABEL_DB, STRICT_LABEL_TABLE, value_column=LABEL)
    label_null_summary = label_non_null_by_date(
        STRICT_LABEL_DB,
        STRICT_LABEL_TABLE,
        LABEL,
        tail_dates=[mature_label_max, SCORE_END],
    )
    payload = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "pre2026_strict_pit_input_materialization",
        "feature_source": {
            "db_path": str(FEATURE_SRC_DB),
            "table": FEATURE_SRC_TABLE,
            "sha256": sha256_file(FEATURE_SRC_DB),
        },
        "label_source": {
            "db_path": str(LABEL_SRC_DB),
            "table": LABEL_SRC_TABLE,
            "sha256": sha256_file(LABEL_SRC_DB),
        },
        "strict_feature_asset": {
            "db_path": str(STRICT_FEATURE_DB),
            "table": STRICT_FEATURE_TABLE,
            "sha256": sha256_file(STRICT_FEATURE_DB),
            "quality": feature_quality,
        },
        "strict_label_asset": {
            "db_path": str(STRICT_LABEL_DB),
            "table": STRICT_LABEL_TABLE,
            "sha256": sha256_file(STRICT_LABEL_DB),
            "quality": label_quality,
        },
        "strict_pit_contract": {
            "score_start": SCORE_START,
            "score_end": SCORE_END,
            "mature_label_max_for_10d": mature_label_max,
            "late_2025_label_payload_null_after": mature_label_max,
            "no_2026_feature_rows_read": True,
            "no_2026_label_payload_retained": True,
            "duckdb_only": True,
            "no_bj": True,
        },
        "label_null_summary": label_null_summary,
    }
    write_json(PRECHECK_JSON, payload)
    log_progress("preflight_materialization_done")
    return payload


def table_quality(db_path: Path, table: str, *, value_column: str | None) -> dict[str, Any]:
    value_probe_sql = (
        f"sum(case when {quote_ident(value_column)} is null then 1 else 0 end) AS null_value_count,"
        if value_column
        else "0 AS null_value_count,"
    )
    with duckdb.connect(str(db_path), read_only=True) as conn:
        latest_date = conn.execute(f"SELECT max(trade_date) FROM {quote_ident(table)}").fetchone()[0]
        row = conn.execute(
            f"""
            SELECT
              count(*) AS row_count,
              count(distinct stock_code) AS stock_count,
              min(trade_date) AS min_trade_date,
              max(trade_date) AS max_trade_date,
              count(distinct trade_date) AS trade_days,
              sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
              {value_probe_sql}
              (
                SELECT count(*)
                FROM (
                  SELECT trade_date, stock_code, count(*) AS c
                  FROM {quote_ident(table)}
                  GROUP BY 1,2
                  HAVING c > 1
                )
              ) AS duplicate_key_groups,
              sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
              count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {quote_ident(table)}
            """,
            [latest_date, latest_date],
        ).fetchdf().iloc[0].to_dict()
    return {
        key: int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
        for key, value in row.items()
    }


def label_non_null_by_date(
    db_path: Path,
    table: str,
    label_col: str,
    *,
    tail_dates: list[str],
) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        rows = []
        for trade_date in tail_dates:
            values = conn.execute(
                f"""
                SELECT
                  ? AS trade_date,
                  count(*) AS row_count,
                  sum(case when {quote_ident(label_col)} is not null then 1 else 0 end) AS non_null_label_rows
                FROM {quote_ident(table)}
                WHERE trade_date = ?
                """,
                [trade_date, trade_date],
            ).fetchone()
            rows.append(
                {
                    "trade_date": str(values[0]),
                    "row_count": int(values[1]),
                    "non_null_label_rows": int(values[2] or 0),
                }
            )
    return {"probe_dates": rows}


def build_windows() -> list[Any]:
    windows = build_rolling_windows(
        data_start=TRAIN_HISTORY_START,
        first_test=SCORE_START,
        final_test=SCORE_END,
        train_years=TRAIN_YEARS,
        test_months=TEST_MONTHS,
        step_months=STEP_MONTHS,
        embargo_days=EMBARGO_DAYS,
        train_mode="fixed",
    )
    write_json(
        FOLD_DEFINITIONS_JSON,
        {
            "generated_at": now_iso(),
            "actor": "model-agent",
            "label": LABEL,
            "fold_count": len(windows),
            "train_history_start": TRAIN_HISTORY_START,
            "score_start": SCORE_START,
            "score_end": SCORE_END,
            "train_years": TRAIN_YEARS,
            "test_months": TEST_MONTHS,
            "step_months": STEP_MONTHS,
            "embargo_days": EMBARGO_DAYS,
            "folds": [asdict(window) for window in windows],
        },
    )
    return windows


def load_existing_summary() -> dict[int, dict[str, Any]]:
    if not FOLD_SUMMARY_CSV.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with FOLD_SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows[int(row["fold"])] = row
            except Exception:
                continue
    return rows


def write_fold_summary(rows: list[dict[str, Any]]) -> None:
    FOLD_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with FOLD_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_training(*, resume: bool) -> list[dict[str, Any]]:
    bindings = {
        ENV_MODEL_FEATURE_DUCKDB: str(STRICT_FEATURE_DB),
        ENV_MODEL_FEATURE_DUCKDB_TABLE: STRICT_FEATURE_TABLE,
        ENV_MODEL_LABEL_DUCKDB: str(STRICT_LABEL_DB),
        ENV_MODEL_LABEL_DUCKDB_TABLE: STRICT_LABEL_TABLE,
    }
    config = FoldFeatureSelectionConfig(
        label=LABEL,
        top_n=FEATURE_TOP_N,
        min_abs_ic=MIN_ABS_IC,
        max_missing_ratio=MAX_MISSING_RATIO,
        folds=FEATURE_IC_FOLDS,
        score_output_dir=str(FOLD_SELECTION_DIR),
    )
    windows = build_windows()
    previous = load_existing_summary() if resume else {}
    summary_rows: list[dict[str, Any]] = [previous[key] for key in sorted(previous.keys()) if key in previous]
    with temporary_env(bindings):
        selector = build_fold_feature_selection_fn(
            data_file_url=str(DATA_DIR),
            config=config,
            feature_source=MODEL_FEATURE_MODE_SPLIT,
        )
        for window in windows:
            prediction_path = FOLD_PREDICTION_DIR / f"fold{window.fold:02d}.parquet"
            if resume and prediction_path.exists() and window.fold in previous:
                continue
            selected_features = selector(window)
            metrics = train_one_fold_with_ai(
                window,
                data_file_url=str(DATA_DIR),
                label=LABEL,
                model_type=MODEL_TYPE,
                selected_features=selected_features,
                prediction_output_path=prediction_path,
                feature_source=MODEL_FEATURE_MODE_SPLIT,
            )
            row = {
                "fold": window.fold,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                **metrics,
            }
            summary_rows = [r for r in summary_rows if int(r["fold"]) != window.fold]
            summary_rows.append(row)
            summary_rows.sort(key=lambda item: int(item["fold"]))
            write_fold_summary(summary_rows)
    return summary_rows


def load_fold_predictions() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(FOLD_PREDICTION_DIR.glob("fold*.parquet")):
        frame = pd.read_parquet(path)
        fold_name = path.stem.replace("fold", "")
        frame["fold_id"] = int(fold_name)
        parts.append(frame)
    if not parts:
        raise RuntimeError("no fold prediction parquet files were produced")
    merged = pd.concat(parts, ignore_index=True)
    merged["trade_date"] = merged["trade_date"].astype(str)
    merged["stock_code"] = merged["stock_code"].astype(str)
    merged = merged[(merged["trade_date"] >= SCORE_START) & (merged["trade_date"] <= SCORE_END)].copy()
    merged = merged.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    merged["score_source"] = "pre2026_strict_pit_oof_10d_20260730"
    merged["candidate_id"] = "research_pre2026_strict_pit_oof_10d_20260730"
    merged["created_at"] = now_iso()
    return merged


def merge_to_duckdb(frame: pd.DataFrame) -> dict[str, Any]:
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(EXPERIMENT_DB)) as conn:
        conn.register("oof_frame", frame)
        conn.execute(f"CREATE OR REPLACE TABLE {quote_ident(EXPERIMENT_TABLE)} AS SELECT * FROM oof_frame")
    return table_quality(EXPERIMENT_DB, EXPERIMENT_TABLE, value_column="pred_prob")


def pred_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["trade_date", "stock_code", "pred_prob"]].itertuples(index=False):
        digest.update(f"{row.trade_date}|{row.stock_code}|{float(row.pred_prob):.17g}\n".encode("utf-8"))
    return digest.hexdigest()


def daily_eval(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["pred_prob", LABEL]).copy()
    if work.empty:
        return pd.DataFrame()
    work["pred_rank"] = work.groupby("trade_date")["pred_prob"].rank(method="average", ascending=False)
    work["label_rank"] = work.groupby("trade_date")[LABEL].rank(method="average", ascending=False)
    rows: list[dict[str, Any]] = []
    for trade_date, group in work.groupby("trade_date"):
        group = group.copy()
        rank_ic = group["pred_prob"].rank().corr(group[LABEL].rank())
        rows.append(
            {
                "trade_date": str(trade_date),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else None,
                "top1": float(group.nsmallest(1, "pred_rank")[LABEL].mean()),
                "top3": float(group.nsmallest(3, "pred_rank")[LABEL].mean()),
                "top5": float(group.nsmallest(5, "pred_rank")[LABEL].mean()),
                "top10": float(group.nsmallest(10, "pred_rank")[LABEL].mean()),
                "top20": float(group.nsmallest(20, "pred_rank")[LABEL].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        return {
            "trade_days": 0,
            "rank_ic": None,
            "top1": None,
            "top3": None,
            "top5": None,
            "top10": None,
            "top20": None,
            "rank_ic_positive_ratio": None,
        }
    return {
        "trade_days": int(len(daily)),
        "rank_ic": safe_mean(daily["rank_ic"]),
        "top1": safe_mean(daily["top1"]),
        "top3": safe_mean(daily["top3"]),
        "top5": safe_mean(daily["top5"]),
        "top10": safe_mean(daily["top10"]),
        "top20": safe_mean(daily["top20"]),
        "rank_ic_positive_ratio": float((daily["rank_ic"] > 0).mean()) if len(daily) else None,
        "min_trade_date": str(daily["trade_date"].min()),
        "max_trade_date": str(daily["trade_date"].max()),
    }


def safe_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def yearly_eval(daily: pd.DataFrame) -> list[dict[str, Any]]:
    if daily.empty:
        return []
    rows: list[dict[str, Any]] = []
    daily = daily.copy()
    daily["year"] = daily["trade_date"].astype(str).str[:4]
    for year, group in daily.groupby("year"):
        summary = summarize_daily(group.drop(columns=["year"]))
        summary["year"] = str(year)
        rows.append(summary)
    return rows


def build_evaluation(frame: pd.DataFrame, mature_label_max: str) -> dict[str, Any]:
    eval_frame = frame[frame["trade_date"] <= mature_label_max].copy()
    daily = daily_eval(eval_frame)
    recent63 = daily.tail(63)
    recent20 = daily.tail(20)
    yearly = yearly_eval(daily)
    daily_csv = REPORT_DIR / "oof_daily_eval.csv"
    yearly_csv = REPORT_DIR / "oof_yearly_eval.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(yearly_csv, index=False, encoding="utf-8-sig")
    return {
        "mature_label_max_for_eval": mature_label_max,
        "full": summarize_daily(daily),
        "recent63": summarize_daily(recent63),
        "recent20": summarize_daily(recent20),
        "yearly": yearly,
        "daily_csv": str(daily_csv),
        "yearly_csv": str(yearly_csv),
    }


def build_hash_inventory(
    *,
    precheck: dict[str, Any],
    quality: dict[str, Any],
    frame: pd.DataFrame,
) -> dict[str, Any]:
    inventory = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "pre2026_strict_pit_oof_10d_hash_inventory",
        "inputs": {
            "feature_source_file": {
                "path": str(FEATURE_SRC_DB),
                "sha256": sha256_file(FEATURE_SRC_DB),
            },
            "label_source_file": {
                "path": str(LABEL_SRC_DB),
                "sha256": sha256_file(LABEL_SRC_DB),
            },
            "strict_feature_file": {
                "path": str(STRICT_FEATURE_DB),
                "sha256": sha256_file(STRICT_FEATURE_DB),
            },
            "strict_label_file": {
                "path": str(STRICT_LABEL_DB),
                "sha256": sha256_file(STRICT_LABEL_DB),
            },
            "feature_source_table": precheck["feature_source"],
            "label_source_table": precheck["label_source"],
        },
        "outputs": {
            "candidate_db": {
                "path": str(EXPERIMENT_DB),
                "sha256": sha256_file(EXPERIMENT_DB),
            },
            "candidate_table": {
                "table": EXPERIMENT_TABLE,
                "quality": quality,
                "pred_prob_sha256": pred_sha256(frame),
            },
        },
    }
    write_json(HASH_INVENTORY_JSON, inventory)
    return inventory


def build_manifest(
    *,
    precheck: dict[str, Any],
    quality: dict[str, Any],
    evaluation: dict[str, Any],
    hash_inventory: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "actor": "model-agent",
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "governance_status": "research_only_strict_pit_oof_candidate_pending_audit",
        "source_type": "duckdb_table",
        "db_path": str(EXPERIMENT_DB),
        "table": EXPERIMENT_TABLE,
        "candidate_id": "research_pre2026_strict_pit_oof_10d_20260730",
        "label": LABEL,
        "coverage": {
            "min_trade_date": SCORE_START,
            "max_trade_date": SCORE_END,
            "quality": quality,
        },
        "strict_pit_contract": {
            "score_window": {"start": SCORE_START, "end": SCORE_END},
            "selection_window_max": "20251231",
            "validation_window_opened": False,
            "mature_label_max_for_eval": precheck["strict_pit_contract"]["mature_label_max_for_10d"],
            "train_and_selection_use_2026_plus": False,
            "duckdb_only": True,
            "no_bj": True,
            "explicit_qfq": True,
            "no_fallback": True,
            "walk_forward": True,
            "quarterly_test_windows": True,
        },
        "fold_definition_path": str(FOLD_DEFINITIONS_JSON),
        "report_json": str(REPORT_JSON),
        "report_md": str(REPORT_MD),
        "hash_inventory": str(HASH_INVENTORY_JSON),
        "evaluation": evaluation,
        "input_assets": {
            "feature_duckdb": str(STRICT_FEATURE_DB),
            "feature_table": STRICT_FEATURE_TABLE,
            "label_duckdb": str(STRICT_LABEL_DB),
            "label_table": STRICT_LABEL_TABLE,
            "feature_source_sha256": precheck["feature_source"]["sha256"],
            "label_source_sha256": precheck["label_source"]["sha256"],
            "strict_feature_sha256": hash_inventory["inputs"]["strict_feature_file"]["sha256"],
            "strict_label_sha256": hash_inventory["inputs"]["strict_label_file"]["sha256"],
        },
        "boundaries": {
            "research_only": True,
            "approved_for_l5": False,
            "allow_next_layer_continue": False,
            "no_formal_change": True,
            "no_production_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    write_json(MANIFEST_JSON, manifest)
    return manifest


def build_report(
    *,
    precheck: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    quality: dict[str, Any],
    evaluation: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "research_only_pre2026_strict_pit_oof_l4_asset_build",
        "status": "completed",
        "label": LABEL,
        "candidate_id": manifest["candidate_id"],
        "feature_input": {
            "db_path": str(STRICT_FEATURE_DB),
            "table": STRICT_FEATURE_TABLE,
        },
        "label_input": {
            "db_path": str(STRICT_LABEL_DB),
            "table": STRICT_LABEL_TABLE,
        },
        "folds": summary_rows,
        "quality": quality,
        "evaluation": evaluation,
        "strict_pit_contract": manifest["strict_pit_contract"],
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "boundaries": manifest["boundaries"],
    }
    write_json(REPORT_JSON, report)

    lines = [
        "# pre-2026 严格 PIT/OOF 10D research-only 资产构建报告",
        "",
        "## 当前结论",
        "",
        "- 已构建新的 `pre2026_strict_pit` 10D research-only L4 历史评分资产。",
        "- 历史评分覆盖 `20220606-20251231`，训练与候选选择均未使用 `2026` 及以后信息。",
        "- 输入为 DuckDB-only、no-BJ、显式 qfq、no fallback；未改 active formal、未改 production。",
        f"- 10D 可监督成熟标签评价截止日：`{precheck['strict_pit_contract']['mature_label_max_for_10d']}`。",
        "",
        "## 构建方式",
        "",
        f"- 训练历史起点：`{TRAIN_HISTORY_START}`",
        f"- 评分覆盖起点：`{SCORE_START}`",
        f"- 评分覆盖终点：`{SCORE_END}`",
        f"- 固定训练窗：`{TRAIN_YEARS}` 年",
        f"- 测试窗：`{TEST_MONTHS}` 个月",
        f"- 步长：`{STEP_MONTHS}` 个月",
        f"- embargo：`{EMBARGO_DAYS}` 天",
        f"- 每折 train-only 特征筛选：Top `{FEATURE_TOP_N}`",
        "",
        "## 质量",
        "",
        f"- 行数：`{quality['row_count']}`",
        f"- 股票数：`{quality['stock_count']}`",
        f"- 日期范围：`{quality['min_trade_date']}` 到 `{quality['max_trade_date']}`",
        f"- 交易日数：`{quality['trade_days']}`",
        f"- 最新日行数 / 股票数：`{quality['latest_day_rows']}` / `{quality['latest_day_stocks']}`",
        f"- `.BJ` 行数：`{quality['bj_rows']}`",
        f"- 重复键：`{quality['duplicate_key_groups']}`",
        "",
        "## 评价摘要",
        "",
        f"- Full RankIC：`{evaluation['full']['rank_ic']}`",
        f"- Full Top1 / Top3 / Top5：`{evaluation['full']['top1']}` / `{evaluation['full']['top3']}` / `{evaluation['full']['top5']}`",
        f"- Recent63 Top5：`{evaluation['recent63']['top5']}`",
        f"- Recent20 Top5：`{evaluation['recent20']['top5']}`",
        "",
        "## 边界",
        "",
        "- research-only。",
        "- 未改 active formal / production。",
        "- 未写 approved_for_l5。",
        "- 未生成策略信号，未跑策略回测。",
        "- `2026` 验证切片仍关闭，留待策略侧冻结唯一候选后一次性打开。",
        "",
        "## 证据路径",
        "",
        f"- manifest：`{MANIFEST_JSON}`",
        f"- 报告 JSON：`{REPORT_JSON}`",
        f"- 报告 MD：`{REPORT_MD}`",
        f"- handoff JSON：`{HANDOFF_JSON}`",
        f"- hash inventory：`{HASH_INVENTORY_JSON}`",
        f"- fold 定义：`{FOLD_DEFINITIONS_JSON}`",
    ]
    write_markdown(REPORT_MD, lines)


def build_handoff() -> None:
    handoff = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "task_id": "research-only-pre2026-strict-pit-oof-l4-asset-build-20260730",
        "status": "completed_pending_audit",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "candidate_manifest": str(MANIFEST_JSON),
        "report_json": str(REPORT_JSON),
        "hash_inventory": str(HASH_INVENTORY_JSON),
        "fold_definitions": str(FOLD_DEFINITIONS_JSON),
        "boundaries": {
            "research_only": True,
            "approved_for_l5": False,
            "no_formal_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    write_json(HANDOFF_JSON, handoff)
    write_markdown(
        HANDOFF_MD,
        [
            "# 审计交接说明",
            "",
            "- 本次交付为 research-only pre-2026 strict PIT/OOF 10D L4 历史评分资产。",
            "- 当前仅可进入 audit-agent 只读复核，不得直接进入 formal、approved_for_l5、production 或策略验证放行。",
            f"- manifest：`{MANIFEST_JSON}`",
            f"- 报告：`{REPORT_JSON}`",
            f"- hash inventory：`{HASH_INVENTORY_JSON}`",
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a research-only strict pre-2026 PIT/OOF 10D L4 asset.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing fold predictions and summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    precheck = materialize_strict_input_assets()
    summary_rows = run_training(resume=args.resume)
    frame = load_fold_predictions()
    quality = merge_to_duckdb(frame)
    evaluation = build_evaluation(frame, mature_label_max=precheck["strict_pit_contract"]["mature_label_max_for_10d"])
    hash_inventory = build_hash_inventory(precheck=precheck, quality=quality, frame=frame)
    manifest = build_manifest(
        precheck=precheck,
        quality=quality,
        evaluation=evaluation,
        hash_inventory=hash_inventory,
    )
    build_report(
        precheck=precheck,
        summary_rows=summary_rows,
        quality=quality,
        evaluation=evaluation,
        manifest=manifest,
    )
    build_handoff()
    # Drop a lightweight prediction manifest for local consistency with other research assets.
    write_prediction_manifest(
        REPORT_DIR / "prediction_manifest.json",
        {
            "label": LABEL,
            "prediction_mode": "duckdb_research_only",
            "prediction_db": str(EXPERIMENT_DB),
            "prediction_table": EXPERIMENT_TABLE,
            "row_count": int(quality["row_count"]),
            "approval_status": "research_only_not_approved_for_l4_or_l5",
            "ready_for_audit_review": True,
            "allow_next_layer_continue": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed_pending_audit",
                "report_json": str(REPORT_JSON),
                "manifest": str(MANIFEST_JSON),
                "candidate_db": str(EXPERIMENT_DB),
                "candidate_table": EXPERIMENT_TABLE,
                "ready_for_audit_review": True,
                "allow_next_layer_continue": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

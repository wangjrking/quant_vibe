from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[5]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_front_combo_reaudit_20260720"
EXPERIMENT_DIR = DATA_DIR / "experimental_assets" / "model-agent" / "l4_predictions"
FEATURE_INPUT_DB = (
    DATA_DIR
    / "runtime"
    / "agent_workspaces"
    / "factor-agent"
    / "work"
    / "l3_target_date_delivery_20260717_candidate_attempt3_20260719"
    / "l3_feature_candidate_20260717.duckdb"
)
ACTIVE_FEATURE_REFERENCE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
LABEL_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_label_current.duckdb"
LABEL_TABLE = "prod_l3_prediction_label_parts_current"
START_DATE = "20220606"
FIT_END = "20251231"
HOLDOUT_START = "20260101"
RUN_DATE_TAG = "20260720"
TASK_ID = "model-5d-10d-front-combo-research-candidates-remediation-20260721"
LABEL_KEYS = ("5d", "10d")
FORMAL_MANIFESTS = {
    "1d": MAIN / "config" / "prediction_manifests" / "executable_1d_open_return_l4_formal_20260619.json",
    "3d": MAIN / "config" / "prediction_manifests" / "executable_3d_open_return_l4_formal_20260617.json",
    "5d": MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json",
    "10d": MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json",
}
LABEL_NAMES = {
    "1d": "executable_1d_open_return",
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
    "10d": "executable_10d_open_return",
}
STRICT_RAW_FEATURES = ["amount", "vol"]
STRICT_MODEL_FEATURES = [
    "pred_1d_rank",
    "pred_3d_rank",
    "pred_5d_rank",
    "pred_10d_rank",
    "pred_1d_gap",
    "pred_3d_gap",
    "pred_5d_gap",
    "pred_10d_gap",
    "rank_consensus_mean",
    "rank_consensus_min",
    "rank_consensus_std",
    "amount_rank",
    "vol_rank",
]
POOL_THRESHOLDS = [0.90, 0.95, 0.97]
BETA5_VALUES = [0.0, 0.01, 0.02, 0.03, 0.04]
BETA3_VALUES = [-0.02, -0.01, 0.0, 0.01, 0.02]
RECENT_WINDOWS = [20, 63, 126]
TOP_K = [1, 3, 5, 10, 20]
TRAIN_POOL_THRESHOLD = min(POOL_THRESHOLDS)

sys.path.insert(0, str(MAIN))
from prediction_manifest import load_prediction_source_manifest  # noqa: E402


@dataclass(frozen=True)
class CandidateSpec:
    label_key: str
    table: str
    db_path: Path
    candidate_id: str
    manifest_path: Path


CANDIDATE_SPECS = {
    "5d": CandidateSpec(
        label_key="5d",
        table="stock_predict_data_model_agent_four_year_5d_front_combo_strict_trainonly_20260720_research",
        db_path=EXPERIMENT_DIR / "l4_5d_front_combo_strict_trainonly_20260720.duckdb",
        candidate_id="research_5d_front_combo_strict_trainonly_nofallback_20260720",
        manifest_path=REPORT_DIR / "research_candidate_manifest_5d_front_combo_strict_20260720.json",
    ),
    "10d": CandidateSpec(
        label_key="10d",
        table="stock_predict_data_model_agent_four_year_10d_front_combo_strict_trainonly_20260720_research",
        db_path=EXPERIMENT_DIR / "l4_10d_front_combo_strict_trainonly_20260720.duckdb",
        candidate_id="research_10d_front_combo_strict_trainonly_nofallback_20260720",
        manifest_path=REPORT_DIR / "research_candidate_manifest_10d_front_combo_strict_20260720.json",
    ),
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 220,
    "learning_rate": 0.035,
    "max_depth": 3,
    "min_child_weight": 30,
    "subsample": 0.78,
    "colsample_bytree": 0.85,
    "reg_lambda": 14.0,
    "tree_method": "hist",
    "n_jobs": 4,
    "eval_metric": "logloss",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def json_ready(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(item) for item in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def validate_required_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = {col: int(frame[col].isna().sum()) for col in columns if col in frame.columns and int(frame[col].isna().sum()) > 0}
    absent = sorted(set(columns) - set(frame.columns))
    if missing or absent:
        details = {"context": context, "missing_null_counts": missing, "absent_columns": absent}
        raise ValueError(f"fail_closed_required_inputs:{json.dumps(details, ensure_ascii=False, sort_keys=True)}")


def selection_sort_key(train_summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(train_summary["train_objective"]),
        float(train_summary["train_top3"]),
        float(train_summary["train_top5"]),
        float(train_summary["train_rank_ic"]),
    )


def build_train_pool(frame: pd.DataFrame, label_key: str, target_top_k: int, pool_threshold: float) -> pd.DataFrame:
    work = add_target_flag(frame, label_key, top_k=target_top_k)
    date_mask = work["trade_date"].astype(str).le(FIT_END)
    rank_mask = pd.to_numeric(work[f"pred_{label_key}_rank"], errors="raise").ge(pool_threshold)
    return work.loc[date_mask & rank_mask].copy()


def score_key_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["trade_date", "stock_code", "pred_prob"]].itertuples(index=False):
        digest.update(f"{row.trade_date}|{row.stock_code}|{float(row.pred_prob):.12g}\n".encode("utf-8"))
    return digest.hexdigest()


def key_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["trade_date", "stock_code"]].itertuples(index=False):
        digest.update(f"{row.trade_date}|{row.stock_code}\n".encode("utf-8"))
    return digest.hexdigest()


def table_fingerprint(db_path: Path, table: str) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        schema_df = con.execute(f"DESCRIBE {quote(table)}").fetchdf()
        has_pred_prob = "pred_prob" in set(schema_df["column_name"].astype(str))
        max_trade_date = con.execute(f"SELECT max(trade_date) FROM {quote(table)}").fetchone()[0]
        pred_prob_null_sql = (
            "sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,"
            if has_pred_prob
            else "CAST(NULL AS BIGINT) AS null_pred_prob,"
        )
        summary_df = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
                {pred_prob_null_sql}
                (
                    SELECT count(*)
                    FROM (
                        SELECT trade_date, stock_code, count(*) c
                        FROM {quote(table)}
                        GROUP BY 1, 2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {quote(table)}
            """,
            [max_trade_date, max_trade_date],
        ).fetchdf()
        latest_df = con.execute(
            f"""
            SELECT trade_date, stock_code
            FROM {quote(table)}
            WHERE trade_date = ?
            ORDER BY trade_date, stock_code
            """,
            [max_trade_date],
        ).fetchdf()
    schema_norm = [
        {"column_name": str(row["column_name"]), "column_type": str(row["column_type"])}
        for _, row in schema_df.iterrows()
    ]
    key_domain = summary_df.iloc[0].to_dict()
    key_domain = {key: (to_float(value) if not isinstance(value, str) else value) for key, value in key_domain.items()}
    return {
        "db_path": str(db_path),
        "table": table,
        "schema": schema_norm,
        "schema_fingerprint_sha256": sha256_bytes(stable_json_bytes(schema_norm)),
        "key_domain": key_domain,
        "key_domain_fingerprint_sha256": sha256_bytes(stable_json_bytes(key_domain)),
        "latest_day_key_sha256": key_sha256(latest_df),
    }


def formal_sources() -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for key, manifest_path in FORMAL_MANIFESTS.items():
        manifest = load_prediction_source_manifest(manifest_path, require_approved=True, allow_legacy=False)
        db_path = Path(manifest["db_path"])
        sources[key] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "db_path": str(db_path),
            "db_sha256": file_sha256(db_path),
            "table": manifest["table"],
            "table_fingerprint": table_fingerprint(db_path, manifest["table"]),
        }
    return sources


def read_formal_scores(sources: dict[str, dict[str, Any]], with_labels: bool) -> pd.DataFrame:
    with duckdb.connect(database=":memory:") as con:
        aliases = {"1d": "one", "3d": "three", "5d": "five", "10d": "ten"}
        for key, alias in aliases.items():
            con.execute(f"ATTACH '{Path(sources[key]['db_path']).as_posix()}' AS {alias}db (READ_ONLY)")
        if with_labels:
            con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS labels (READ_ONLY)")
            label_cols = ",\n                ".join(
                f"labels.{LABEL_NAMES[key]} AS label_{key}" for key in FORMAL_MANIFESTS
            )
            label_join = f"JOIN labels.{LABEL_TABLE} labels USING (trade_date, stock_code)"
            select_labels = f",\n                {label_cols}"
        else:
            label_join = ""
            select_labels = ""
        query = f"""
            SELECT
                onep.trade_date,
                onep.stock_code,
                onep.pred_prob AS pred_1d,
                threep.pred_prob AS pred_3d,
                fivep.pred_prob AS pred_5d,
                tenp.pred_prob AS pred_10d
                {select_labels}
            FROM onedb.{quote(sources['1d']['table'])} onep
            JOIN threedb.{quote(sources['3d']['table'])} threep USING (trade_date, stock_code)
            JOIN fivedb.{quote(sources['5d']['table'])} fivep USING (trade_date, stock_code)
            JOIN tendb.{quote(sources['10d']['table'])} tenp USING (trade_date, stock_code)
            {label_join}
            WHERE onep.trade_date >= ?
              AND onep.stock_code NOT LIKE '%.BJ'
              AND onep.pred_prob IS NOT NULL
              AND threep.pred_prob IS NOT NULL
              AND fivep.pred_prob IS NOT NULL
              AND tenp.pred_prob IS NOT NULL
            ORDER BY onep.trade_date, onep.stock_code
        """
        return con.execute(query, [START_DATE]).fetchdf()


def load_feature_slice(feature_db_path: Path) -> pd.DataFrame:
    cols = ", ".join(["trade_date", "stock_code", *STRICT_RAW_FEATURES])
    with duckdb.connect(str(feature_db_path), read_only=True) as con:
        return con.execute(
            f"""
            SELECT {cols}
            FROM {FEATURE_TABLE}
            WHERE trade_date >= ?
              AND stock_code NOT LIKE '%.BJ'
            ORDER BY trade_date, stock_code
            """,
            [START_DATE],
        ).fetchdf()


def add_rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["pred_1d", "pred_3d", "pred_5d", "pred_10d"]:
        out[f"{col}_rank"] = out.groupby("trade_date")[col].rank(method="average", pct=True)
    return out


def prepare_base(feature_db_path: Path, sources: dict[str, dict[str, Any]], with_labels: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    score_frame = add_rank_columns(read_formal_scores(sources, with_labels=with_labels))
    feature_frame = load_feature_slice(feature_db_path)
    merged = score_frame.merge(
        feature_frame,
        on=["trade_date", "stock_code"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    missing_feature_rows = int((merged["_merge"] == "left_only").sum())
    if missing_feature_rows:
        sample = (
            merged.loc[merged["_merge"] == "left_only", ["trade_date", "stock_code"]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            "fail_closed_feature_key_domain:"
            + json.dumps(
                {
                    "feature_db_path": str(feature_db_path),
                    "missing_feature_rows": missing_feature_rows,
                    "sample_missing_keys": sample,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    merged = merged.drop(columns=["_merge"])
    validate_required_columns(merged, STRICT_RAW_FEATURES, "strict_raw_features")
    for col in STRICT_RAW_FEATURES:
        merged[f"{col}_rank"] = merged.groupby("trade_date")[col].rank(method="average", pct=True)
    rank_cols = [f"pred_{key}_rank" for key in FORMAL_MANIFESTS]
    for key in FORMAL_MANIFESTS:
        others = [col for col in rank_cols if col != f"pred_{key}_rank"]
        merged[f"pred_{key}_gap"] = merged[f"pred_{key}_rank"] - merged[others].mean(axis=1)
    merged["rank_consensus_mean"] = merged[rank_cols].mean(axis=1)
    merged["rank_consensus_min"] = merged[rank_cols].min(axis=1)
    merged["rank_consensus_std"] = merged[rank_cols].std(axis=1)
    for col in STRICT_MODEL_FEATURES:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    validate_required_columns(merged, STRICT_MODEL_FEATURES, "strict_model_features")
    merged[STRICT_MODEL_FEATURES] = merged[STRICT_MODEL_FEATURES].astype("float32")
    evidence = {
        "feature_db_path": str(feature_db_path),
        "score_rows": int(len(score_frame)),
        "feature_rows": int(len(feature_frame)),
        "joined_rows": int(len(merged)),
        "strict_raw_features": list(STRICT_RAW_FEATURES),
        "strict_model_features": list(STRICT_MODEL_FEATURES),
        "missing_feature_rows": missing_feature_rows,
        "raw_feature_null_counts": {col: int(merged[col].isna().sum()) for col in STRICT_RAW_FEATURES},
        "model_feature_null_counts": {col: int(merged[col].isna().sum()) for col in STRICT_MODEL_FEATURES},
        "min_trade_date": str(merged["trade_date"].min()),
        "max_trade_date": str(merged["trade_date"].max()),
    }
    return merged, evidence


def add_target_flag(frame: pd.DataFrame, label_key: str, top_k: int) -> pd.DataFrame:
    label_col = f"label_{label_key}"
    work = frame[frame[label_col].notna()].copy()
    work["_label_rank_desc"] = work.groupby("trade_date")[label_col].rank(method="first", ascending=False)
    work[f"target_top{top_k}"] = (work["_label_rank_desc"] <= top_k).astype("int8")
    return work


def train_classifier(train_pool: pd.DataFrame, target_col: str, label_key: str, seed_offset: int) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    positives = int(train_pool[target_col].sum())
    negatives = int(len(train_pool) - positives)
    scale = max(1.0, negatives / max(1, positives))
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = float(scale)
    params["random_state"] = 20260720 + seed_offset + {"5d": 5, "10d": 10}[label_key]
    model = xgb.XGBClassifier(**params)
    model.fit(train_pool[STRICT_MODEL_FEATURES], train_pool[target_col])
    return model, {
        "target_col": target_col,
        "train_rows": int(len(train_pool)),
        "positive_rows": positives,
        "negative_rows": negatives,
        "scale_pos_weight": float(scale),
        "fit_min_trade_date": str(train_pool["trade_date"].min()) if len(train_pool) else None,
        "fit_max_trade_date": str(train_pool["trade_date"].max()) if len(train_pool) else None,
        "feature_count": len(STRICT_MODEL_FEATURES),
        "model_params": model.get_params(),
    }


def daily_eval(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group[group[label_col].notna()]
        if len(group) < max(TOP_K):
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        row: dict[str, Any] = {
            "trade_date": str(trade_date),
            "rows": int(len(ordered)),
            "rank_ic": to_float(ordered[score_col].corr(ordered[label_col], method="spearman")),
        }
        for k in TOP_K:
            row[f"top{k}"] = to_float(ordered.head(k)[label_col].mean())
        rows.append(row)
    daily = pd.DataFrame(rows)
    daily["year"] = daily["trade_date"].str.slice(0, 4)
    daily["month"] = daily["trade_date"].str.slice(0, 6)
    return daily


def metrics(daily: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "trade_days": int(len(daily)),
        "rank_ic": to_float(daily["rank_ic"].mean()) if len(daily) else None,
        "rank_ic_positive_ratio": to_float((daily["rank_ic"] > 0).mean()) if len(daily) else None,
    }
    for k in TOP_K:
        out[f"top{k}"] = to_float(daily[f"top{k}"].mean()) if len(daily) else None
        out[f"top{k}_positive_ratio"] = to_float((daily[f"top{k}"] > 0).mean()) if len(daily) else None
    return out


def period_stability(daily: pd.DataFrame, by: str) -> dict[str, Any]:
    metric_cols = ["rank_ic", *[f"top{k}" for k in TOP_K]]
    grouped = daily[[by, *metric_cols]].groupby(by, sort=True)[metric_cols].mean().reset_index()
    out: dict[str, Any] = {"periods": int(len(grouped))}
    for col in metric_cols:
        out[f"{col}_positive_period_ratio"] = to_float((grouped[col] > 0).mean()) if len(grouped) else None
        out[f"min_{col}"] = to_float(grouped[col].min()) if len(grouped) else None
        worst = grouped.sort_values(col, ascending=True).head(1)
        out[f"worst_{col}_{by}"] = str(worst.iloc[0][by]) if not worst.empty else None
    return out


def metrics_pack(daily: pd.DataFrame) -> dict[str, Any]:
    out = {
        "full": metrics(daily),
        "train": metrics(daily[daily["trade_date"] <= FIT_END]),
        "holdout": metrics(daily[daily["trade_date"] >= HOLDOUT_START]),
        "recent20": metrics(daily.tail(20)),
        "recent63": metrics(daily.tail(63)),
        "recent126": metrics(daily.tail(126)),
        "annual_stability": period_stability(daily, "year"),
        "monthly_stability": period_stability(daily, "month"),
        "eval_min_trade_date": str(daily["trade_date"].min()) if len(daily) else None,
        "eval_max_trade_date": str(daily["trade_date"].max()) if len(daily) else None,
        "eval_trade_days": int(len(daily)),
    }
    return out


def delta_block(candidate: dict[str, Any], baseline: dict[str, Any], block: str) -> dict[str, float | None]:
    return {
        metric: to_float((candidate[block].get(metric) or 0.0) - (baseline[block].get(metric) or 0.0))
        for metric in ["rank_ic", "top1", "top3", "top5", "top10", "top20"]
    }


def build_candidate_score(frame: pd.DataFrame, label_key: str, p5_rank: pd.Series, p3_rank: pd.Series, pool_threshold: float, beta5: float, beta3: float) -> pd.Series:
    base_col = f"pred_{label_key}_rank"
    score = frame[base_col].astype("float64").copy()
    mask = frame[base_col] >= pool_threshold
    score.loc[mask] = (
        frame.loc[mask, base_col]
        + beta5 * (p5_rank.loc[mask] - 0.5)
        + beta3 * (p3_rank.loc[mask] - 0.5)
    )
    return score


def evaluate_scan_row(
    frame: pd.DataFrame,
    label_key: str,
    baseline_metrics: dict[str, Any],
    p5_rank: pd.Series,
    p3_rank: pd.Series,
    pool_threshold: float,
    beta5: float,
    beta3: float,
) -> dict[str, Any]:
    work = frame.copy()
    work["candidate_score"] = build_candidate_score(work, label_key, p5_rank, p3_rank, pool_threshold, beta5, beta3)
    daily = daily_eval(work, "candidate_score", f"label_{label_key}")
    packed = metrics_pack(daily)
    deltas = {
        block: delta_block(packed, baseline_metrics, block)
        for block in ["full", "train", "holdout", "recent20", "recent63", "recent126"]
    }
    train_summary = {
        "train_objective": float(
            2.0 * (deltas["train"].get("top3") or -1.0)
            + 1.2 * (deltas["train"].get("top5") or -1.0)
            + 0.8 * (deltas["train"].get("top1") or -1.0)
            + 0.05 * (deltas["train"].get("rank_ic") or -1.0)
        ),
        "train_top3": float(deltas["train"].get("top3") or -1.0),
        "train_top5": float(deltas["train"].get("top5") or -1.0),
        "train_rank_ic": float(deltas["train"].get("rank_ic") or -1.0),
    }
    row = {
        "label_key": label_key,
        "pool_threshold": float(pool_threshold),
        "beta5": float(beta5),
        "beta3": float(beta3),
        "metrics": packed,
        "deltas": deltas,
        "train_selection": train_summary,
        "selection_sort_key": list(selection_sort_key(train_summary)),
    }
    return row


def freeze_only_gate(row: dict[str, Any]) -> dict[str, Any]:
    deltas = row["deltas"]
    reasons: list[str] = []
    if (deltas["full"].get("rank_ic") or 0.0) < -0.0015:
        reasons.append("full_rank_ic_delta_below_-0_0015")
    if (deltas["full"].get("top5") or 0.0) <= 0.0:
        reasons.append("full_top5_delta_non_positive")
    if (deltas["holdout"].get("top5") or 0.0) <= 0.0:
        reasons.append("holdout_top5_delta_non_positive")
    if (deltas["recent63"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent63_top5_delta_non_positive")
    if (deltas["recent20"].get("top5") or 0.0) <= 0.0:
        reasons.append("recent20_top5_delta_non_positive")
    return {"passed": not reasons, "failed_reasons": reasons}


def select_best_row(scan_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(scan_rows, key=lambda item: selection_sort_key(item["train_selection"]), reverse=True)[0]


def materialize_candidate(
    full_frame: pd.DataFrame,
    label_key: str,
    top5_model: xgb.XGBClassifier,
    top3_model: xgb.XGBClassifier,
    selected: dict[str, Any],
) -> dict[str, Any]:
    spec = CANDIDATE_SPECS[label_key]
    spec.db_path.parent.mkdir(parents=True, exist_ok=True)
    frame = full_frame.copy()
    frame["_p5"] = top5_model.predict_proba(frame[STRICT_MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p3"] = top3_model.predict_proba(frame[STRICT_MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p5_rank"] = frame.groupby("trade_date")["_p5"].rank(method="average", pct=True)
    frame["_p3_rank"] = frame.groupby("trade_date")["_p3"].rank(method="average", pct=True)
    out = frame[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = build_candidate_score(
        frame,
        label_key,
        frame["_p5_rank"],
        frame["_p3_rank"],
        selected["pool_threshold"],
        selected["beta5"],
        selected["beta3"],
    ).astype("float64")
    out["score_source"] = spec.candidate_id
    out["model_label"] = LABEL_NAMES[label_key]
    out["created_at"] = now_iso()
    prewrite_score_sha256 = score_key_sha256(out.sort_values(["trade_date", "stock_code"]).reset_index(drop=True))
    latest_trade_date = str(out["trade_date"].max())
    latest_keys = out.loc[out["trade_date"] == latest_trade_date, ["trade_date", "stock_code"]].sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    latest_key_sha = key_sha256(latest_keys)
    with duckdb.connect(str(spec.db_path)) as con:
        con.register("candidate_df", out)
        con.execute(f"CREATE OR REPLACE TABLE {quote(spec.table)} AS SELECT * FROM candidate_df")
        readback = con.execute(
            f"SELECT trade_date, stock_code, pred_prob FROM {quote(spec.table)} ORDER BY trade_date, stock_code"
        ).fetchdf()
        quality_row = con.execute(
            f"""
            SELECT
                count(*) AS row_count,
                min(trade_date) AS min_trade_date,
                max(trade_date) AS max_trade_date,
                count(distinct trade_date) AS trade_days,
                count(distinct stock_code) AS stock_count,
                sum(case when stock_code like '%.BJ' then 1 else 0 end) AS bj_rows,
                sum(case when pred_prob is null then 1 else 0 end) AS null_pred_prob,
                (
                    SELECT count(*)
                    FROM (
                        SELECT trade_date, stock_code, count(*) c
                        FROM {quote(spec.table)}
                        GROUP BY 1, 2
                        HAVING c > 1
                    )
                ) AS duplicate_key_groups,
                sum(case when trade_date = ? then 1 else 0 end) AS latest_day_rows,
                count(distinct case when trade_date = ? then stock_code end) AS latest_day_stocks
            FROM {quote(spec.table)}
            """,
            [latest_trade_date, latest_trade_date],
        ).fetchdf().iloc[0].to_dict()
        latest_readback = con.execute(
            f"SELECT trade_date, stock_code FROM {quote(spec.table)} WHERE trade_date = ? ORDER BY trade_date, stock_code",
            [latest_trade_date],
        ).fetchdf()
    postwrite_score_sha256 = score_key_sha256(readback)
    latest_day_consistency = latest_key_sha == key_sha256(latest_readback)
    quality = {key: (to_float(value) if not isinstance(value, str) else value) for key, value in quality_row.items()}
    return {
        "candidate_spec": {
            "label_key": label_key,
            "candidate_id": spec.candidate_id,
            "db_path": str(spec.db_path),
            "table": spec.table,
        },
        "quality": quality,
        "prewrite_pred_prob_sha256": prewrite_score_sha256,
        "postwrite_pred_prob_sha256": postwrite_score_sha256,
        "formula_row_consistency_passed": prewrite_score_sha256 == postwrite_score_sha256,
        "latest_trade_date": latest_trade_date,
        "latest_day_key_sha256": latest_key_sha,
        "latest_day_key_consistency_passed": latest_day_consistency,
        "candidate_db_sha256": file_sha256(spec.db_path),
        "candidate_table_fingerprint": table_fingerprint(spec.db_path, spec.table),
    }


def build_candidate_manifest(
    label_key: str,
    materialized: dict[str, Any],
    selected: dict[str, Any],
    train_artifacts: dict[str, Any],
    config_path: Path,
    hash_inventory_path: Path,
    feature_input_fingerprint: dict[str, Any],
    active_feature_reference_fingerprint: dict[str, Any],
    label_fingerprint: dict[str, Any],
    formal_inputs: dict[str, Any],
) -> dict[str, Any]:
    spec = CANDIDATE_SPECS[label_key]
    manifest = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "actor": "model-agent",
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_pending_reaudit",
        "governance_status": "restricted_p1_not_for_strategy_or_formal_until_reaudit",
        "source_type": "duckdb_table",
        "db_path": str(spec.db_path),
        "table": spec.table,
        "label": LABEL_NAMES[label_key],
        "candidate_id": spec.candidate_id,
        "selection_method": "strict_train_only_front_combo_scan",
        "selection_freeze_windows": {
            "train_end": FIT_END,
            "holdout_start": HOLDOUT_START,
            "recent_windows": RECENT_WINDOWS,
            "selection_uses_holdout_or_recent": False,
        },
        "score_formula": (
            f"pred_{label_key}_rank + {selected['beta5']}*(front_top5_rank-0.5) + "
            f"{selected['beta3']}*(front_top3_rank-0.5), applied only when pred_{label_key}_rank >= {selected['pool_threshold']}"
        ),
        "pool_threshold": selected["pool_threshold"],
        "beta5": selected["beta5"],
        "beta3": selected["beta3"],
        "train_selection": selected["train_selection"],
        "post_freeze_gate": freeze_only_gate(selected),
        "quality": materialized["quality"],
        "pred_prob_sha256": materialized["postwrite_pred_prob_sha256"],
        "formula_row_consistency_passed": materialized["formula_row_consistency_passed"],
        "latest_day_key_consistency_passed": materialized["latest_day_key_consistency_passed"],
        "lineage": {
            "feature_input": {
                "path": str(FEATURE_INPUT_DB),
                "table": FEATURE_TABLE,
                "fingerprint": feature_input_fingerprint,
            },
            "active_feature_reference": {
                "path": str(ACTIVE_FEATURE_REFERENCE_DB),
                "table": FEATURE_TABLE,
                "fingerprint": active_feature_reference_fingerprint,
            },
            "label_input": {
                "path": str(LABEL_DB),
                "table": LABEL_TABLE,
                "fingerprint": label_fingerprint,
            },
            "formal_inputs": formal_inputs,
            "train_artifacts": train_artifacts,
            "config_path": str(config_path),
            "hash_inventory_path": str(hash_inventory_path),
        },
        "restrictions": {
            "strategy_observation_scoring_forbidden": True,
            "signal_forbidden": True,
            "backtest_forbidden": True,
            "formal_forbidden": True,
            "approved_for_l5_forbidden": True,
            "production_forbidden": True,
            "l5_to_l8_forbidden": True,
        },
        "frozen_forward_boundary": "20260722",
    }
    spec.manifest_path.write_text(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "run_date": "2026-07-20",
        "feature_input_db": str(FEATURE_INPUT_DB),
        "active_feature_reference_db": str(ACTIVE_FEATURE_REFERENCE_DB),
        "label_db": str(LABEL_DB),
        "feature_table": FEATURE_TABLE,
        "label_table": LABEL_TABLE,
        "strict_raw_features": STRICT_RAW_FEATURES,
        "strict_model_features": STRICT_MODEL_FEATURES,
        "start_date": START_DATE,
        "fit_end": FIT_END,
        "holdout_start": HOLDOUT_START,
        "pool_thresholds": POOL_THRESHOLDS,
        "beta5_values": BETA5_VALUES,
        "beta3_values": BETA3_VALUES,
        "xgb_params": XGB_PARAMS,
        "selection_sort_key_fields": ["train_objective", "train_top3", "train_top5", "train_rank_ic"],
        "selection_uses_only_train_window": True,
        "holdout_recent_used_only_after_freeze": True,
        "no_fallback_fill_allowed": True,
        "no_bj": True,
        "duckdb_only": True,
        "explicit_qfq_contract_respected": True,
        "frozen_forward_boundary": "20260722",
    }
    config_path = REPORT_DIR / "strict_front_combo_config_20260720.json"
    config_path.write_text(json.dumps(json_ready(config), ensure_ascii=False, indent=2), encoding="utf-8")

    formal_input_info = formal_sources()
    labeled_base, labeled_evidence = prepare_base(FEATURE_INPUT_DB, formal_input_info, with_labels=True)
    full_base, full_evidence = prepare_base(FEATURE_INPUT_DB, formal_input_info, with_labels=False)

    feature_input_fingerprint = table_fingerprint(FEATURE_INPUT_DB, FEATURE_TABLE)
    active_feature_reference_fingerprint = table_fingerprint(ACTIVE_FEATURE_REFERENCE_DB, FEATURE_TABLE)
    label_fingerprint = table_fingerprint(LABEL_DB, LABEL_TABLE)

    combined_scan_rows: list[dict[str, Any]] = []
    scan_outputs: dict[str, str] = {}
    selection_evidence: dict[str, Any] = {
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "selection_sort_key_fields": config["selection_sort_key_fields"],
        "selection_uses_only_train_window": True,
        "holdout_recent_used_only_after_freeze": True,
        "labels": {},
    }
    summary: dict[str, Any] = {
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "status": "ready_for_audit_review",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "approval_status": "research_only_pending_reaudit",
        "feature_input": str(FEATURE_INPUT_DB),
        "active_feature_reference": str(ACTIVE_FEATURE_REFERENCE_DB),
        "label_input": str(LABEL_DB),
        "artifacts": {},
        "labels": {},
    }

    hash_inventory: dict[str, Any] = {
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "feature_input_db_path": str(FEATURE_INPUT_DB),
        "feature_input_db_sha256": file_sha256(FEATURE_INPUT_DB),
        "active_feature_reference_db_path": str(ACTIVE_FEATURE_REFERENCE_DB),
        "active_feature_reference_db_sha256": file_sha256(ACTIVE_FEATURE_REFERENCE_DB),
        "label_db_path": str(LABEL_DB),
        "label_db_sha256": file_sha256(LABEL_DB),
        "feature_input_fingerprint": feature_input_fingerprint,
        "active_feature_reference_fingerprint": active_feature_reference_fingerprint,
        "label_fingerprint": label_fingerprint,
        "formal_inputs": formal_input_info,
        "candidates": {},
    }

    for label_key in LABEL_KEYS:
        label_frame = labeled_base[["trade_date", "stock_code", *STRICT_MODEL_FEATURES, f"label_{label_key}"]].copy()
        label_frame = label_frame[label_frame[f"label_{label_key}"].notna()].copy()
        train_pool_top5 = build_train_pool(label_frame, label_key, target_top_k=5, pool_threshold=TRAIN_POOL_THRESHOLD)
        train_pool_top3 = build_train_pool(label_frame, label_key, target_top_k=3, pool_threshold=TRAIN_POOL_THRESHOLD)
        top5_model, top5_info = train_classifier(train_pool_top5, "target_top5", label_key, seed_offset=100)
        top3_model, top3_info = train_classifier(train_pool_top3, "target_top3", label_key, seed_offset=200)

        p5 = top5_model.predict_proba(label_frame[STRICT_MODEL_FEATURES])[:, 1].astype("float32")
        p3 = top3_model.predict_proba(label_frame[STRICT_MODEL_FEATURES])[:, 1].astype("float32")
        label_frame["_p5_rank"] = pd.Series(p5, index=label_frame.index).groupby(label_frame["trade_date"]).rank(method="average", pct=True)
        label_frame["_p3_rank"] = pd.Series(p3, index=label_frame.index).groupby(label_frame["trade_date"]).rank(method="average", pct=True)

        baseline_daily = daily_eval(label_frame, f"pred_{label_key}_rank", f"label_{label_key}")
        baseline_metrics = metrics_pack(baseline_daily)
        scan_rows: list[dict[str, Any]] = []
        for pool_threshold in POOL_THRESHOLDS:
            for beta5 in BETA5_VALUES:
                for beta3 in BETA3_VALUES:
                    row = evaluate_scan_row(
                        label_frame,
                        label_key,
                        baseline_metrics,
                        label_frame["_p5_rank"],
                        label_frame["_p3_rank"],
                        pool_threshold,
                        beta5,
                        beta3,
                    )
                    row["post_freeze_gate"] = freeze_only_gate(row)
                    scan_rows.append(row)

        selected = select_best_row(scan_rows)
        scan_csv = REPORT_DIR / f"{label_key}_front_combo_strict_trainonly_scan_20260720.csv"
        flat_scan = []
        for row in scan_rows:
            flat_scan.append(
                {
                    "label_key": row["label_key"],
                    "pool_threshold": row["pool_threshold"],
                    "beta5": row["beta5"],
                    "beta3": row["beta3"],
                    "train_objective": row["train_selection"]["train_objective"],
                    "train_top3": row["train_selection"]["train_top3"],
                    "train_top5": row["train_selection"]["train_top5"],
                    "train_rank_ic": row["train_selection"]["train_rank_ic"],
                    "full_top3_delta": row["deltas"]["full"]["top3"],
                    "full_top5_delta": row["deltas"]["full"]["top5"],
                    "holdout_top5_delta": row["deltas"]["holdout"]["top5"],
                    "recent63_top5_delta": row["deltas"]["recent63"]["top5"],
                    "recent20_top5_delta": row["deltas"]["recent20"]["top5"],
                    "post_freeze_gate_passed": row["post_freeze_gate"]["passed"],
                    "post_freeze_gate_failed_reasons": ";".join(row["post_freeze_gate"]["failed_reasons"]),
                    "selection_sort_key": json.dumps(row["selection_sort_key"]),
                }
            )
        pd.DataFrame(flat_scan).to_csv(scan_csv, index=False, encoding="utf-8-sig")
        combined_scan_rows.extend(flat_scan)
        scan_outputs[label_key] = str(scan_csv)

        materialized = materialize_candidate(full_base, label_key, top5_model, top3_model, selected)
        train_artifacts = {
            "top5_train_info": top5_info,
            "top3_train_info": top3_info,
            "strict_model_features": STRICT_MODEL_FEATURES,
            "top5_model_sha256": sha256_bytes(top5_model.get_booster().save_raw(raw_format="json")),
            "top3_model_sha256": sha256_bytes(top3_model.get_booster().save_raw(raw_format="json")),
        }
        hash_inventory["candidates"][label_key] = {
            "candidate_db_path": str(materialized["candidate_spec"]["db_path"]),
            "candidate_db_sha256": materialized["candidate_db_sha256"],
            "candidate_table": materialized["candidate_spec"]["table"],
            "candidate_table_fingerprint": materialized["candidate_table_fingerprint"],
            "candidate_manifest_path": str(CANDIDATE_SPECS[label_key].manifest_path),
        }
        manifest = build_candidate_manifest(
            label_key,
            materialized,
            selected,
            train_artifacts,
            config_path,
            REPORT_DIR / "hash_inventory_20260720.json",
            feature_input_fingerprint,
            active_feature_reference_fingerprint,
            label_fingerprint,
            {k: {
                "manifest_path": v["manifest_path"],
                "manifest_sha256": v["manifest_sha256"],
                "db_path": v["db_path"],
                "db_sha256": v["db_sha256"],
                "table": v["table"],
                "table_fingerprint": v["table_fingerprint"],
            } for k, v in formal_input_info.items()},
        )

        selection_evidence["labels"][label_key] = {
            "selected_pool_threshold": selected["pool_threshold"],
            "selected_beta5": selected["beta5"],
            "selected_beta3": selected["beta3"],
            "selection_sort_key": selected["selection_sort_key"],
            "train_selection": selected["train_selection"],
            "post_freeze_gate": selected["post_freeze_gate"],
            "scan_csv": str(scan_csv),
        }
        summary["labels"][label_key] = {
            "candidate_id": CANDIDATE_SPECS[label_key].candidate_id,
            "candidate_db_path": str(materialized["candidate_spec"]["db_path"]),
            "candidate_table": materialized["candidate_spec"]["table"],
            "candidate_manifest_path": str(CANDIDATE_SPECS[label_key].manifest_path),
            "quality": materialized["quality"],
            "formula_row_consistency_passed": materialized["formula_row_consistency_passed"],
            "latest_day_key_consistency_passed": materialized["latest_day_key_consistency_passed"],
            "selected": {
                "pool_threshold": selected["pool_threshold"],
                "beta5": selected["beta5"],
                "beta3": selected["beta3"],
                "train_selection": selected["train_selection"],
                "post_freeze_gate": selected["post_freeze_gate"],
                "deltas": selected["deltas"],
            },
        }

    combined_scan_csv = REPORT_DIR / "front_combo_strict_trainonly_combined_scan_20260720.csv"
    pd.DataFrame(combined_scan_rows).to_csv(combined_scan_csv, index=False, encoding="utf-8-sig")
    selection_evidence["combined_scan_csv"] = str(combined_scan_csv)
    selection_evidence_path = REPORT_DIR / "selection_evidence_20260720.json"
    selection_evidence_path.write_text(json.dumps(json_ready(selection_evidence), ensure_ascii=False, indent=2), encoding="utf-8")

    hash_inventory_path = REPORT_DIR / "hash_inventory_20260720.json"
    hash_inventory_path.write_text(json.dumps(json_ready(hash_inventory), ensure_ascii=False, indent=2), encoding="utf-8")

    summary["artifacts"] = {
        "config_path": str(config_path),
        "selection_evidence_path": str(selection_evidence_path),
        "hash_inventory_path": str(hash_inventory_path),
        "combined_scan_csv": str(combined_scan_csv),
        "per_label_scan_csv": scan_outputs,
    }
    summary["input_evidence"] = {
        "labeled_base": labeled_evidence,
        "full_base": full_evidence,
    }
    summary_path = REPORT_DIR / "front_combo_reaudit_summary_20260720.json"
    summary_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    handoff = {
        "generated_at": now_iso(),
        "task_id": TASK_ID,
        "status": "ready_for_audit_review",
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "approval_status": "research_only_pending_reaudit",
        "summary_path": str(summary_path),
        "selection_evidence_path": str(selection_evidence_path),
        "hash_inventory_path": str(hash_inventory_path),
        "candidate_manifests": {key: str(spec.manifest_path) for key, spec in CANDIDATE_SPECS.items()},
        "restrictions": [
            "strategy_observation_scoring_forbidden",
            "signal_forbidden",
            "backtest_forbidden",
            "formal_forbidden",
            "approved_for_l5_forbidden",
            "production_forbidden",
            "l5_to_l8_forbidden",
        ],
    }
    handoff_path = REPORT_DIR / "audit_handoff_20260720.json"
    handoff_path.write_text(json.dumps(json_ready(handoff), ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# 5D / 10D front combo research-only 整改重建报告",
        "",
        f"- 任务编号：`{TASK_ID}`",
        "- 当前日期：`2026-07-20`",
        "- 状态：`ready_for_audit_review=true`，`allow_next_layer_continue=false`",
        "- 最远状态：`research_only_pending_reaudit`",
        "",
        "## 本轮整改",
        "",
        "- 候选选择改为严格 `train-only`；`holdout/recent63/recent20` 仅在冻结后评价，不参与筛选、排序或淘汰。",
        "- 移除 `0.5` 静默兜底；当前候选重建只允许完整键域和完整严格特征输入，缺键或关键字段缺失即 fail-closed。",
        "- 使用新 `candidate_id`、新 DuckDB 文件和新 manifest，不覆盖旧 restricted 资产。",
        "- 血缘中显式冻结 candidate feature 输入、active feature 参考、active label、四份 formal 输入及其 SHA256 / 表级指纹。",
        "",
        "## 候选结果",
        "",
        "| 标签 | candidate_id | 日期范围 | 最新日 | 行数 | 股票数 | .BJ | null_pred_prob | duplicate | post-freeze gate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for label_key in LABEL_KEYS:
        item = summary["labels"][label_key]
        quality = item["quality"]
        gate = item["selected"]["post_freeze_gate"]
        md_lines.append(
            "| {label} | `{cid}` | `{min_d}-{max_d}` | `{latest}` | {rows:.0f} | {stocks:.0f} | {bj:.0f} | {nulls:.0f} | {dup:.0f} | `{gate}` |".format(
                label=label_key,
                cid=item["candidate_id"],
                min_d=quality["min_trade_date"],
                max_d=quality["max_trade_date"],
                latest=quality["max_trade_date"],
                rows=quality["row_count"] or 0.0,
                stocks=quality["latest_day_stocks"] or 0.0,
                bj=quality["bj_rows"] or 0.0,
                nulls=quality["null_pred_prob"] or 0.0,
                dup=quality["duplicate_key_groups"] or 0.0,
                gate="passed" if gate["passed"] else ";".join(gate["failed_reasons"]),
            )
        )
    md_lines.extend(
        [
            "",
            "## 证据",
            "",
            f"- 配置：`{config_path}`",
            f"- 选择证据：`{selection_evidence_path}`",
            f"- Hash inventory：`{hash_inventory_path}`",
            f"- 汇总：`{summary_path}`",
            f"- 审计交接：`{handoff_path}`",
            "",
            "## 边界",
            "",
            "- 未修改 active formal / production manifest。",
            "- 未训练生产模型。",
            "- 未生成策略信号。",
            "- 未跑策略回测。",
            "- 未推进 L5-L8。",
        ]
    )
    (REPORT_DIR / "front_combo_reaudit_summary_20260720.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary_path": str(summary_path), "handoff_path": str(handoff_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

from adjustment_semantics import (
    NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS,
    NAKED_MARKET_PRICE_COLUMNS,
    default_adjustment_semantics,
    default_market_field_semantics,
    explicit_qfq_column_name,
)
from l3_duckdb_sync import GTJA_QFQ_COLUMN_MAP
from model_asset_route import (
    resolve_model_feature_duckdb_path,
    resolve_model_feature_duckdb_table,
    resolve_model_label_duckdb_path,
    resolve_model_label_duckdb_table,
)


ROOT = Path(__file__).resolve().parents[2]
QUANT_DIR = ROOT / "quant"
DATA_DIR = QUANT_DIR / "data_file"
MAIN_DIR = QUANT_DIR / "main"
DUCKDB_DIR = DATA_DIR / "production_assets" / "duckdb"
MANIFEST_DIR = MAIN_DIR / "config" / "prediction_manifests"
PRODUCTION_ASSET_REGISTRY_PATH = DATA_DIR / "asset_registry" / "production_assets.json"
TARGET_DATE = os.environ.get("MODEL_FORMAL_L4_TARGET_DATE", "20260629")
REPORT_DIR = DATA_DIR / "reports" / f"model_agent_formal_incremental_l4_{TARGET_DATE}"

MARKET_DB_PATH_IN_MANIFEST = "../../../data_file/production_assets/duckdb/l2_stock_daily_data.duckdb"

FORMAL_TABLES = {
    "1d": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate",
    "3d": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate",
    "5d": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate",
    "10d": "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate",
    "prod_5d": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
}

FORMAL_DUCKDB_FILES = {
    "1d": "l4_executable_1d_open_return_formal.duckdb",
    "3d": "l4_executable_3d_open_return_formal.duckdb",
    "5d": "l4_executable_5d_open_return_formal.duckdb",
    "10d": "l4_executable_10d_open_return_formal.duckdb",
    "prod_5d": "l4_prod_liq_prime_one_v20260612_compat_formal.duckdb",
}

MODEL_SPECS = {
    "1d": {
        "label": "executable_1d_open_return",
        "manifest": "executable_1d_open_return_l4_formal_20260619.json",
        "target_table": FORMAL_TABLES["1d"],
    },
    "3d": {
        "label": "executable_3d_open_return",
        "manifest": "executable_3d_open_return_l4_formal_20260617.json",
        "target_table": FORMAL_TABLES["3d"],
    },
    "5d": {
        "label": "executable_5d_open_return",
        "manifest": "executable_5d_open_return_l4_formal_20260620.json",
        "target_table": FORMAL_TABLES["5d"],
        "compat_manifest": "prod_liq_prime_one_v20260612_l4_formal.json",
        "compat_table": FORMAL_TABLES["prod_5d"],
    },
    "10d": {
        "label": "executable_10d_open_return",
        "manifest": "executable_10d_open_return_l4_formal_20260617.json",
        "target_table": FORMAL_TABLES["10d"],
    },
}

LEGACY_FRONT_ADJUSTED_ALIAS_BASES = set(NAKED_MARKET_PRICE_COLUMNS) | set(NAKED_FRONT_ADJUSTED_INDICATOR_COLUMNS) | {
    "macdsignal",
    "macdhist",
    "macd_dea",
    "macd_dif",
    "rsi",
    "tema",
    "dema",
    "dema_10",
    "t3",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_root_path(raw_path: str) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else (ROOT / path).resolve()


def reject_nonproduction_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    forbidden = ("research", "legacy", "fallback", "model_predictions.db", "odb.db")
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"production model binding rejects non-production path: {path}")


def resolve_approved_production_model_binding(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve inference artifacts only from an approved manifest and registry binding."""
    manifest_path = MANIFEST_DIR / str(spec["manifest"])
    manifest = load_manifest(str(spec["manifest"]))
    if manifest.get("approval_status") != "approved_for_l5" or manifest.get("asset_role") != "l4_formal_prediction_asset":
        raise RuntimeError(f"manifest is not an approved formal L4 asset: {manifest_path}")

    required = (
        "production_model_asset_id",
        "production_model_metadata_path",
        "production_model_path",
        "production_model_metadata_sha256",
        "production_model_sha256",
    )
    missing = [field for field in required if not str(manifest.get(field) or "").strip()]
    if missing:
        raise RuntimeError(
            f"manifest lacks explicit approved production model binding ({', '.join(missing)}): {manifest_path}"
        )

    registry = json.loads(PRODUCTION_ASSET_REGISTRY_PATH.read_text(encoding="utf-8"))
    asset_id = str(manifest["production_model_asset_id"])
    registry_asset = next(
        (item for item in registry.get("assets", []) if item.get("asset_id") == asset_id),
        None,
    )
    if not registry_asset or registry_asset.get("track") != "production" or not registry_asset.get("allowed_for_main_workflow"):
        raise RuntimeError(f"production model asset is not approved in registry: {asset_id}")

    metadata_path = resolve_root_path(str(manifest["production_model_metadata_path"]))
    model_path = resolve_root_path(str(manifest["production_model_path"]))
    reject_nonproduction_path(metadata_path)
    reject_nonproduction_path(model_path)
    if not metadata_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"approved production model binding is missing files: {asset_id}")
    if file_sha256(metadata_path) != str(manifest["production_model_metadata_sha256"]):
        raise RuntimeError(f"production metadata hash mismatch: {asset_id}")
    if file_sha256(model_path) != str(manifest["production_model_sha256"]):
        raise RuntimeError(f"production model hash mismatch: {asset_id}")

    return {
        "asset_id": asset_id,
        "registry_asset_path": registry_asset.get("asset_path"),
        "metadata_path": metadata_path,
        "model_path": model_path,
        "metadata_sha256": str(manifest["production_model_metadata_sha256"]),
        "model_sha256": str(manifest["production_model_sha256"]),
    }


def quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def resolve_model_path(metadata_path: Path, metadata: dict[str, Any]) -> Path:
    raw = Path(str(metadata["model_path"]))
    if raw.is_absolute():
        return raw
    from_meta = (metadata_path.parent / raw).resolve()
    if from_meta.exists():
        return from_meta
    fallback = (MAIN_DIR / raw).resolve()
    return fallback


def factor_input_asset() -> tuple[Path, str]:
    duckdb_path = resolve_model_feature_duckdb_path(DATA_DIR, require_exists=True)
    table = resolve_model_feature_duckdb_table(DATA_DIR)
    if not table:
        raise RuntimeError("active L3 feature DuckDB table is missing")
    return duckdb_path, table


def duckdb_input_provenance(path: Path, table: str) -> dict[str, str]:
    """Describe the actual L3 DuckDB consumed by this prediction run."""
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    normalized = resolved.as_posix().lower()
    candidate_only = "/runtime/agent_workspaces/" in normalized and "candidate" in resolved.name.lower()
    source_type = "candidate_duckdb_table" if candidate_only else "duckdb_table"
    return {
        "asset": f"{display_path}::{table}",
        "path": str(resolved),
        "table": table,
        "source_type": source_type,
        "input_role": "candidate_only" if candidate_only else "active",
    }


def load_partial_exception_context(
    *,
    factor_path: Path,
    factor_table: str,
    target_date: str,
) -> dict[str, Any] | None:
    """Load an explicitly authorized partial-L3 exception without altering scores."""
    if os.environ.get("MODEL_FORMAL_L4_USER_AUTHORIZED_PARTIAL_EXCEPTION", "").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return None

    raw_handoff_path = os.environ.get("MODEL_FORMAL_L4_PARTIAL_HANDOFF")
    if not raw_handoff_path:
        raise RuntimeError("partial exception requires MODEL_FORMAL_L4_PARTIAL_HANDOFF")
    handoff_path = Path(raw_handoff_path).resolve()
    if not handoff_path.is_file():
        raise RuntimeError(f"partial exception handoff is missing: {handoff_path}")
    handoff = json.loads(handoff_path.read_text(encoding="utf-8-sig"))
    candidate = handoff.get("candidate") or {}
    if not handoff.get("partial_source_not_ready") or not handoff.get("candidate_only"):
        raise RuntimeError("partial exception handoff does not declare a candidate-only source limitation")
    if str(handoff.get("target_trade_date")) != target_date:
        raise RuntimeError("partial exception target date does not match the L4 target date")
    if Path(str(candidate.get("path", ""))).resolve() != factor_path.resolve():
        raise RuntimeError("partial exception candidate path does not match the consumed L3 DuckDB")
    if str(candidate.get("table")) != factor_table:
        raise RuntimeError("partial exception candidate table does not match the consumed L3 table")
    if file_sha256(factor_path) != str(candidate.get("sha256")):
        raise RuntimeError("partial exception candidate SHA256 does not match the consumed L3 DuckDB")

    return {
        "user_authorized_partial_exception": True,
        "partial_source_not_ready": True,
        "active_l3_claimed": False,
        "handoff_path": str(handoff_path),
        "handoff_sha256": file_sha256(handoff_path),
        "source_limitations": handoff.get("source_limitations", {}),
        "candidate": candidate,
        "model_route": os.environ.get("MODEL_FORMAL_L4_AGENT_MODEL_ROUTE", "unspecified"),
    }


def label_input_asset() -> tuple[Path, str]:
    duckdb_path = resolve_model_label_duckdb_path(DATA_DIR, require_exists=True)
    table = resolve_model_label_duckdb_table(DATA_DIR)
    if not table:
        raise RuntimeError("active L3 label DuckDB table is missing")
    return duckdb_path, table


def build_manifest_update_payload(
    *,
    manifest: dict,
    label_key: str,
    stats: dict,
    formula_info: dict,
    archive_name: str,
    target_date: str,
    db_path_in_manifest: str,
) -> dict:
    payload = dict(manifest)
    payload.pop("score_source_usage", None)
    payload.pop("completion_row_share", None)
    payload.update(
        {
            "schema_version": payload.get("schema_version", 1),
            "asset_role": "l4_formal_prediction_asset",
            "approval_status": "approved_for_l5",
            "source_type": "duckdb_table",
            "db_path": db_path_in_manifest,
            "table": FORMAL_TABLES[label_key],
            "market_db_path": MARKET_DB_PATH_IN_MANIFEST,
            "adjustment_semantics": default_adjustment_semantics(),
            "market_field_semantics": default_market_field_semantics(),
            "generated_at": now_iso(),
            "row_count": stats["row_count"],
            "trade_days": stats["trade_days"],
            "stock_count": stats["stock_count"],
            "min_trade_date": stats["min_trade_date"],
            "max_trade_date": stats["max_trade_date"],
            "latest_day_rows": stats["latest_day_rows"],
            "latest_day_stock_count": stats["latest_day_stock_count"],
            "duplicate_keys": stats["duplicate_key_groups"],
            "null_pred_prob": stats["null_pred_prob"],
            "pred_prob_sha256": stats["pred_prob_sha256"],
            "score_formula": formula_info["formula"],
            "formula_sources": formula_info["sources"],
            "latest_incremental_update": {
                "trade_date": target_date,
                "rows": stats["latest_day_rows"],
                "method": "saved_production_model_and_approved_formula_incremental_l4_scoring",
                "production_factor_input": formula_info["factor_input"],
                "label_input": formula_info["label_input"],
                "no_training": True,
                "no_new_research_model": True,
                "no_signal": True,
                "no_backtest": True,
                "report_path": f"../../../data_file/reports/model_agent_formal_incremental_l4_{target_date}/formal_incremental_l4_{target_date}_report.json",
                "user_authorized_partial_exception": bool(formula_info.get("partial_exception")),
            },
            f"previous_manifest_archive_before_incremental_l4_{target_date}": archive_name,
            "lineage": "current_incremental_duckdb_only_savedmodel_completion",
            "lineage_sources": {
                "l3_feature_duckdb_asset": formula_info["factor_input"],
                "l3_feature_source_type": formula_info.get("factor_input_source_type", "duckdb_table"),
                "l3_feature_input_role": formula_info.get("factor_input_role", "active"),
                "user_authorized_partial_exception": bool(formula_info.get("partial_exception")),
                "active_l3_label_duckdb_asset": formula_info["label_input"],
                "current_score_source": "savedmodel_completion",
                "historical_release_lineage_archive": archive_name,
            },
            "notes": (
                f"已按当前 DuckDB 标准链路补齐 {target_date} 的 L4 formal 预测结果；"
                "本次只使用已保存生产模型和已批准 formal 评分公式做增量刷新，"
                "未训练、未调参、未生成交易信号、未运行回测。"
            ),
            "rollback_note": (
                f"如需回滚，恢复 {archive_name} 指向的 manifest；"
                "本次未删除旧表，仅对 active formal 表追加/覆盖目标交易日。"
            ),
        }
    )
    if formula_info.get("partial_exception"):
        payload["partial_input_exception"] = formula_info["partial_exception"]
    if label_key == "prod_5d":
        payload["strategy_id"] = "prod_liq_prime_one_v20260612"
        payload["governance_status"] = "approved_l5_consumable_strategy_specific_compat_manifest_not_default_task"
    return payload


def archive_manifest(name: str) -> str:
    src = MANIFEST_DIR / name
    archive = src.with_name(src.stem + f"_archive_before_{TARGET_DATE}_incremental_l4.json")
    if not archive.exists():
        shutil.copy2(src, archive)
    return archive.name


def load_manifest(name: str) -> dict[str, Any]:
    return json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))


def write_manifest(name: str, payload: dict[str, Any]) -> None:
    (MANIFEST_DIR / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_manifest(path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(MAIN_DIR))
    from prediction_manifest import load_prediction_source_manifest

    source = load_prediction_source_manifest(str(path), require_approved=True, allow_legacy=False)
    return {
        "manifest": str(path.as_posix()),
        "db_path": str(source.get("db_path")),
        "table": str(source.get("table")),
        "approval_status": str(source.get("approval_status")),
        "source_type": str(source.get("source_type")),
    }


def manifest_formula_info(
    manifest: dict[str, Any],
    *,
    factor_input: str,
    label_input: str,
    factor_input_provenance: dict[str, str] | None = None,
    partial_exception: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = factor_input_provenance or {
        "source_type": "duckdb_table",
        "input_role": "active",
    }
    return {
        "formula": "latest_incremental_rows: score = savedmodel_completion",
        "sources": ["savedmodel_completion"],
        "factor_input": factor_input,
        "label_input": label_input,
        "factor_input_source_type": provenance["source_type"],
        "factor_input_role": provenance["input_role"],
        "partial_exception": partial_exception,
    }


def formal_duckdb_path(label_key: str) -> Path:
    return DUCKDB_DIR / FORMAL_DUCKDB_FILES[label_key]


def formal_duckdb_manifest_path(label_key: str) -> str:
    return f"../../../data_file/production_assets/duckdb/{FORMAL_DUCKDB_FILES[label_key]}"


def load_factor_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    factor_table: str,
    trade_date: str,
    columns: list[str],
    allow_savedmodel_qfq_aliases: bool = False,
) -> pd.DataFrame:
    available = {
        str(row[1]) for row in con.execute(f"PRAGMA table_info({quote(factor_table)})").fetchall()
    }
    alias_pairs: list[tuple[str, str]] = []
    if allow_savedmodel_qfq_aliases:
        resolved = resolve_savedmodel_feature_aliases(columns, available)
        selected = resolved["query_columns"]
        missing = resolved["missing"]
        alias_pairs = resolved["alias_pairs"]
    else:
        selected = [column for column in columns if column in available]
        missing = sorted(set(columns) - set(selected))
    if missing:
        raise RuntimeError(f"L3 factor DuckDB table is missing required columns: {missing[:20]}")
    sql = (
        f"SELECT {', '.join(quote(column) for column in selected)} "
        f"FROM {quote(factor_table)} "
        "WHERE trade_date = ? "
        "ORDER BY stock_code"
    )
    frame = con.execute(sql, [trade_date]).fetchdf()
    if frame.empty:
        raise RuntimeError(f"no L3 factor rows found for {trade_date}")
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    for requested_name, source_name in alias_pairs:
        frame[requested_name] = frame[source_name]
    return frame


def resolve_savedmodel_feature_alias(feature_name: str, available_columns: set[str]) -> str | None:
    feature = str(feature_name)
    if feature in available_columns:
        return feature
    qfq_gtja = GTJA_QFQ_COLUMN_MAP.get(feature)
    if qfq_gtja and qfq_gtja in available_columns:
        return qfq_gtja
    if feature in LEGACY_FRONT_ADJUSTED_ALIAS_BASES:
        qfq_name = explicit_qfq_column_name(feature)
        if qfq_name in available_columns:
            return qfq_name
    return None


def resolve_savedmodel_feature_aliases(
    requested_columns: list[str],
    available_columns: set[str],
) -> dict[str, Any]:
    query_columns: list[str] = []
    seen: set[str] = set()
    alias_pairs: list[tuple[str, str]] = []
    missing: list[str] = []

    for column in requested_columns:
        resolved = resolve_savedmodel_feature_alias(column, available_columns)
        if resolved is None:
            missing.append(str(column))
            continue
        if resolved not in seen:
            query_columns.append(resolved)
            seen.add(resolved)
        if resolved != column:
            alias_pairs.append((str(column), str(resolved)))

    return {
        "query_columns": query_columns,
        "alias_pairs": alias_pairs,
        "missing": missing,
    }


def load_label_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    label_table: str,
    label_column: str,
    trade_date: str,
    stock_codes: list[str],
) -> pd.DataFrame:
    available = {
        str(row[1]) for row in con.execute(f"PRAGMA table_info({quote(label_table)})").fetchall()
    }
    if label_column not in available:
        raise RuntimeError(f"L3 label DuckDB table is missing label column: {label_column}")
    sql = (
        f"SELECT trade_date, stock_code, {quote(label_column)} "
        f"FROM {quote(label_table)} "
        "WHERE trade_date = ? "
        "ORDER BY stock_code"
    )
    frame = con.execute(sql, [trade_date]).fetchdf()
    if frame.empty:
        frame = pd.DataFrame(
            {
                "trade_date": [trade_date] * len(stock_codes),
                "stock_code": stock_codes,
                label_column: [np.nan] * len(stock_codes),
            }
        )
    else:
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["stock_code"] = frame["stock_code"].astype(str)
        if len(frame) != len(stock_codes):
            merged = pd.DataFrame({"trade_date": trade_date, "stock_code": stock_codes})
            frame = merged.merge(frame, on=["trade_date", "stock_code"], how="left")
    return frame


def predict_savedmodel_frame(
    con: duckdb.DuckDBPyConnection,
    *,
    factor_table: str,
    trade_date: str,
    metadata_path: Path,
    approved_model_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = approved_model_path or resolve_model_path(metadata_path, metadata)
    features = [str(column) for column in metadata["feature_columns"]]
    frame = load_factor_frame(
        con,
        factor_table=factor_table,
        trade_date=trade_date,
        columns=["trade_date", "stock_code", *features],
        allow_savedmodel_qfq_aliases=True,
    )
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    x = frame[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    pred = np.asarray(booster.inplace_predict(x.to_numpy(copy=False)), dtype="float64")
    scored = frame[["trade_date", "stock_code"]].copy()
    scored["pred_prob"] = pred
    available_columns = {
        str(row[1]) for row in con.execute(f"PRAGMA table_info({quote(factor_table)})").fetchall()
    }
    alias_resolved = resolve_savedmodel_feature_aliases(features, available_columns)
    return scored, {
        "metadata_path": str(metadata_path),
        "model_path": str(model_path),
        "feature_count": len(features),
        "qfq_alias_pairs": alias_resolved["alias_pairs"],
    }


def build_formal_frame(
    *,
    trade_date: str,
    label_column: str,
    scored: pd.DataFrame,
    label_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = scored.merge(
        label_frame[["trade_date", "stock_code", label_column]],
        on=["trade_date", "stock_code"],
        how="left",
    )
    merged["score_source"] = "savedmodel_completion"
    return merged


def write_trade_date_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    trade_date: str,
    frame: pd.DataFrame,
) -> None:
    out = frame.copy()
    con.register("_append_frame", out)
    try:
        table_exists = bool(
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [table],
            ).fetchone()[0]
        )
        if not table_exists:
            con.execute(
                f"CREATE TABLE {quote(table)} AS SELECT * FROM _append_frame WHERE 1 = 0"
            )
        schema = con.execute(f"DESCRIBE {quote(table)}").fetchall()
        schema_columns = [str(row[0]) for row in schema]
        for column in schema_columns:
            if column not in out.columns:
                out[column] = np.nan
        out = out[schema_columns]
        con.unregister("_append_frame")
        con.register("_append_frame", out)
        con.execute(f"DELETE FROM {quote(table)} WHERE trade_date = ?", [trade_date])
        con.execute(f"INSERT INTO {quote(table)} SELECT * FROM _append_frame")
    finally:
        con.unregister("_append_frame")


def table_digest(con: duckdb.DuckDBPyConnection, table: str) -> str:
    digest = hashlib.sha256()
    cursor = con.execute(
        f"""
        SELECT trade_date, stock_code, pred_prob
        FROM {quote(table)}
        ORDER BY trade_date, stock_code
        """
    )
    while True:
        rows = cursor.fetchmany(100_000)
        if not rows:
            break
        for trade_date, stock_code, pred_prob in rows:
            digest.update(str(trade_date).encode("utf-8"))
            digest.update(b"|")
            digest.update(str(stock_code).encode("utf-8"))
            digest.update(b"|")
            digest.update(format(float(pred_prob), ".17g").encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def duckdb_stats(con: duckdb.DuckDBPyConnection, table: str, *, latest_date: str) -> dict[str, Any]:
    row = con.execute(
        f"""
        SELECT
            count(*),
            min(trade_date),
            max(trade_date),
            count(distinct trade_date),
            count(distinct stock_code),
            sum(case when pred_prob is null then 1 else 0 end)
        FROM {quote(table)}
        """
    ).fetchone()
    dup = con.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT trade_date, stock_code, count(*) AS c
            FROM {quote(table)}
            GROUP BY trade_date, stock_code
            HAVING c > 1
        )
        """
    ).fetchone()[0]
    latest = con.execute(
        f"""
        SELECT count(*), count(distinct stock_code),
               sum(case when stock_code like '%.BJ' then 1 else 0 end)
        FROM {quote(table)}
        WHERE trade_date = ?
        """,
        [latest_date],
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
        "latest_day_rows": int(latest[0] or 0),
        "latest_day_stock_count": int(latest[1] or 0),
        "latest_day_bj_rows": int(latest[2] or 0),
        "pred_prob_sha256": table_digest(con, table),
    }


def factor_and_label_status(
    factor_con: duckdb.DuckDBPyConnection,
    label_con: duckdb.DuckDBPyConnection,
    *,
    factor_table: str,
    label_table: str,
    target_date: str,
) -> dict[str, Any]:
    factor_row = factor_con.execute(
        f"""
        SELECT min(trade_date), max(trade_date), count(*), count(distinct trade_date)
        FROM {quote(factor_table)}
        """
    ).fetchone()
    label_row = label_con.execute(
        f"""
        SELECT min(trade_date), max(trade_date), count(*), count(distinct trade_date)
        FROM {quote(label_table)}
        """
    ).fetchone()
    factor_target = factor_con.execute(
        f"""
        SELECT count(*), count(distinct stock_code)
        FROM {quote(factor_table)}
        WHERE trade_date = ?
        """,
        [target_date],
    ).fetchone()
    return {
        "factor_table": factor_table,
        "factor_min_trade_date": str(factor_row[0]),
        "factor_max_trade_date": str(factor_row[1]),
        "factor_rows": int(factor_row[2]),
        "factor_trade_days": int(factor_row[3]),
        "factor_target_rows": int(factor_target[0] or 0),
        "factor_target_stocks": int(factor_target[1] or 0),
        "label_table": label_table,
        "label_min_trade_date": str(label_row[0]),
        "label_max_trade_date": str(label_row[1]),
        "label_rows": int(label_row[2]),
        "label_trade_days": int(label_row[3]),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['target_date']} L4 formal 增量预测刷新报告",
        "",
        "## 当前结论",
        "",
        f"- 已按当前 DuckDB 标准链路补齐 `{report['target_date']}` 的 L4 formal 预测资产。",
        "- 本次只使用已保存生产模型和当前已批准 formal 评分公式。",
        "- 未训练、未调参、未生成交易信号、未运行回测。",
        "",
        "## 输入资产",
        "",
        f"- L3 特征：`{report['inputs']['factor_asset']}`",
        f"- L3 标签：`{report['inputs']['label_asset']}`",
        "",
        "## 输出资产",
        "",
        "| 标签 | manifest | 表 | 最新日期 | 最新日行数 | 最新日股票数 | 重复键 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["outputs"]:
        stats = item["stats"]
        lines.append(
            f"| {item['label_key']} | `{item['manifest']}` | `{item['table']}` | "
            f"{stats['max_trade_date']} | {stats['latest_day_rows']} | {stats['latest_day_stock_count']} | {stats['duplicate_key_groups']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 未训练模型",
            "- 未调参",
            "- 未生成交易信号",
            "- 未运行回测",
            "- 未切换新的 production manifest，只更新当前 formal 资产的增量状态",
            "",
            "## 证据",
            "",
            f"- `{report['report_json_path']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    factor_db_path, factor_table = factor_input_asset()
    label_db_path, label_table = label_input_asset()
    factor_input_provenance = duckdb_input_provenance(factor_db_path, factor_table)
    partial_exception = load_partial_exception_context(
        factor_path=factor_db_path,
        factor_table=factor_table,
        target_date=TARGET_DATE,
    )
    if partial_exception and factor_input_provenance["source_type"] != "candidate_duckdb_table":
        raise RuntimeError("partial exception requires a candidate DuckDB input")
    factor_input = factor_input_provenance["asset"]
    label_input = f"quant/data_file/production_assets/duckdb/{label_db_path.name}::{label_table}"

    outputs: list[dict[str, Any]] = []
    manifest_archives: dict[str, str] = {}
    manifest_validation: list[dict[str, Any]] = []
    scoring_runs: list[dict[str, Any]] = []
    with duckdb.connect(str(factor_db_path), read_only=True) as factor_con, duckdb.connect(
        str(label_db_path), read_only=True
    ) as label_con:
        status = factor_and_label_status(
            factor_con,
            label_con,
            factor_table=factor_table,
            label_table=label_table,
            target_date=TARGET_DATE,
        )
        if status["factor_target_rows"] <= 0:
            raise RuntimeError(f"L3 feature mainline has no rows for target date {TARGET_DATE}")

        for label_key, spec in MODEL_SPECS.items():
            production_binding = resolve_approved_production_model_binding(spec)
            scored, model_meta = predict_savedmodel_frame(
                factor_con,
                factor_table=factor_table,
                trade_date=TARGET_DATE,
                metadata_path=production_binding["metadata_path"],
                approved_model_path=production_binding["model_path"],
            )
            label_frame = load_label_frame(
                label_con,
                label_table=label_table,
                label_column=str(spec["label"]),
                trade_date=TARGET_DATE,
                stock_codes=scored["stock_code"].tolist(),
            )
            formal_frame = build_formal_frame(
                trade_date=TARGET_DATE,
                label_column=str(spec["label"]),
                scored=scored,
                label_frame=label_frame,
            )

            target_db_path = formal_duckdb_path(label_key)
            target_db_path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(target_db_path), read_only=False) as out_con:
                write_trade_date_rows(
                    out_con,
                    table=str(spec["target_table"]),
                    trade_date=TARGET_DATE,
                    frame=formal_frame,
                )
                out_con.commit()
                stats = duckdb_stats(out_con, str(spec["target_table"]), latest_date=TARGET_DATE)

            scoring_runs.append(
                {
                    "label_key": label_key,
                    "label": spec["label"],
                    "metadata_path": model_meta["metadata_path"],
                    "model_path": model_meta["model_path"],
                    "production_model_asset_id": production_binding["asset_id"],
                    "production_model_registry_asset_path": production_binding["registry_asset_path"],
                    "production_model_metadata_sha256": production_binding["metadata_sha256"],
                    "production_model_sha256": production_binding["model_sha256"],
                    "feature_count": model_meta["feature_count"],
                    "qfq_alias_count": len(model_meta["qfq_alias_pairs"]),
                    "qfq_alias_pairs": model_meta["qfq_alias_pairs"],
                    "target_duckdb_path": str(target_db_path),
                    "score_source_usage_for_target_date": {
                        key: int(value)
                        for key, value in formal_frame["score_source"].value_counts(dropna=False).to_dict().items()
                    },
                }
            )

            manifest_name = str(spec["manifest"])
            manifest_payload = load_manifest(manifest_name)
            archive_name = archive_manifest(manifest_name)
            formula_info = manifest_formula_info(
                manifest_payload,
                factor_input=factor_input,
                label_input=label_input,
                factor_input_provenance=factor_input_provenance,
                partial_exception=partial_exception,
            )
            updated = build_manifest_update_payload(
                manifest=manifest_payload,
                label_key=label_key,
                stats=stats,
                formula_info=formula_info,
                archive_name=archive_name,
                target_date=TARGET_DATE,
                db_path_in_manifest=formal_duckdb_manifest_path(label_key),
            )
            write_manifest(manifest_name, updated)
            manifest_archives[label_key] = archive_name
            outputs.append(
                {
                    "label_key": label_key,
                    "manifest": manifest_name,
                    "table": str(spec["target_table"]),
                    "duckdb_path": str(target_db_path),
                    "stats": stats,
                }
            )

        compat_spec = MODEL_SPECS["5d"]
        compat_production_binding = resolve_approved_production_model_binding(compat_spec)
        compat_frame = build_formal_frame(
            trade_date=TARGET_DATE,
            label_column=str(compat_spec["label"]),
            scored=predict_savedmodel_frame(
                factor_con,
                factor_table=factor_table,
                trade_date=TARGET_DATE,
                metadata_path=compat_production_binding["metadata_path"],
                approved_model_path=compat_production_binding["model_path"],
            )[0],
            label_frame=load_label_frame(
                label_con,
                label_table=label_table,
                label_column=str(compat_spec["label"]),
                trade_date=TARGET_DATE,
                stock_codes=load_factor_frame(
                    factor_con,
                    factor_table=factor_table,
                    trade_date=TARGET_DATE,
                    columns=["trade_date", "stock_code"],
                )["stock_code"].tolist(),
            ),
        ).drop(columns=["score_source"], errors="ignore")

        compat_db_path = formal_duckdb_path("prod_5d")
        compat_db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(compat_db_path), read_only=False) as compat_con:
            write_trade_date_rows(
                compat_con,
                table=str(compat_spec["compat_table"]),
                trade_date=TARGET_DATE,
                frame=compat_frame,
            )
            compat_con.commit()
            prod_stats = duckdb_stats(compat_con, FORMAL_TABLES["prod_5d"], latest_date=TARGET_DATE)

        prod_manifest_name = str(compat_spec["compat_manifest"])
        prod_manifest_payload = load_manifest(prod_manifest_name)
        prod_archive = archive_manifest(prod_manifest_name)
        prod_formula_info = manifest_formula_info(
            prod_manifest_payload,
            factor_input=factor_input,
            label_input=label_input,
            factor_input_provenance=factor_input_provenance,
            partial_exception=partial_exception,
        )
        prod_updated = build_manifest_update_payload(
            manifest=prod_manifest_payload,
            label_key="prod_5d",
            stats=prod_stats,
            formula_info=prod_formula_info,
            archive_name=prod_archive,
            target_date=TARGET_DATE,
            db_path_in_manifest=formal_duckdb_manifest_path("prod_5d"),
        )
        write_manifest(prod_manifest_name, prod_updated)
        manifest_archives["prod_5d"] = prod_archive
        outputs.append(
            {
                "label_key": "prod_5d",
                "manifest": prod_manifest_name,
                "table": FORMAL_TABLES["prod_5d"],
                "duckdb_path": str(compat_db_path),
                "stats": prod_stats,
            }
        )

    for item in outputs:
        manifest_validation.append(validate_manifest(MANIFEST_DIR / item["manifest"]))

    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "target_date": TARGET_DATE,
        "runtime": {
            "python_executable": sys.executable,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "duckdb_version": duckdb.__version__,
        },
        "inputs": {
            "factor_asset": factor_input,
            "factor_input_provenance": factor_input_provenance,
            "label_asset": label_input,
            "factor_asset_sha256": file_sha256(factor_db_path),
            "label_asset_sha256": file_sha256(label_db_path),
            "factor_status": status,
            "partial_exception": partial_exception,
        },
        "outputs": outputs,
        "scoring_runs": scoring_runs,
        "contract_fix": {
            "method": "explicit_savedmodel_qfq_alias_bridge",
            "rules": [
                "gtja_alpha*** legacy metadata names map to gtja_alpha***_qfq only when the explicit qfq column exists in active L3",
                "legacy implicit front-adjusted price names open/high/low/close/pre_close map to *_qfq only when the explicit qfq column exists in active L3",
                "legacy implicit front-adjusted technical names such as atr/macd/macdsignal/macdhist/kdj/wr/mfi/rsi/tema/dema/dema_10/t3 map to *_qfq only when the explicit qfq column exists in active L3",
                "unrelated non-qfq fields are not remapped; unresolved names still fail closed"
            ]
        },
        "manifest_archives": manifest_archives,
        "manifest_validation": manifest_validation,
        "ready_for_audit_review": True,
        "allow_next_layer_continue": False,
        "allow_l5_continue": False,
        "closure_status": "pending_audit_for_l4_formal_incremental_refresh",
        "residual_risk": [
            f"L3 标签主线仍只到 20260616；{TARGET_DATE} 仅刷新预测，不写空壳标签，也不把 {TARGET_DATE} 包装为成熟可评价日期。",
            f"{TARGET_DATE} 的 active L3 可用股票范围为 {status['factor_target_rows']} 行 / {status['factor_target_stocks']} 只；若与上一交易日不同，属于上游当日可用股票范围变化，不视为 L4 去重丢失。",
            "本次通过显式 qfq alias 桥接兼容旧 saved-model metadata；后续若要彻底消除该桥接层，仍需单独治理历史模型 metadata 的特征命名。"
        ],
        "boundaries": {
            "no_training": True,
            "no_tuning": True,
            "no_signal": True,
            "no_backtest": True,
            "no_new_research_manifest": True,
        },
    }

    report_json = REPORT_DIR / f"formal_incremental_l4_{TARGET_DATE}_report.json"
    report_md = REPORT_DIR / f"formal_incremental_l4_{TARGET_DATE}_report.md"
    report["report_json_path"] = str(report_json)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

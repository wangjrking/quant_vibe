from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
DATA_DIR = ROOT / "quant" / "data_file"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_four_year_3d5d10d_front_combo_calibrator_20260714"
EXPERIMENT_DB = (
    DATA_DIR
    / "experimental_assets"
    / "model-agent"
    / "l4_predictions"
    / "l4_5d_front_combo_calibrator_20260714.duckdb"
)
TABLE = "stock_predict_data_model_agent_four_year_5d_front_combo_calibrator_20260714_research"
FEATURE_DB = DATA_DIR / "production_assets" / "duckdb" / "l3_feature_current.duckdb"
FEATURE_TABLE = "prod_l3_production_factor_parts_20260625"
TOP5_MODEL = (
    DATA_DIR
    / "experimental_assets"
    / "model-agent"
    / "models"
    / "front_top5_calibrator_3d5d10d_20260714"
    / "front_top5_calibrator_5d_research_20260714.json"
)
TOP3_MODEL = (
    DATA_DIR
    / "experimental_assets"
    / "model-agent"
    / "models"
    / "front_top3_calibrator_3d5d10d_20260714"
    / "front_top3_calibrator_5d_research_20260714.json"
)
SOURCE_REPORT = REPORT_DIR / "front_combo_calibrator_3d5d10d_report.json"

sys.path.insert(0, str(MAIN))
import research_four_year_3d5d10d_front_top5_calibrator_20260714 as ftop  # noqa: E402


POOL_THRESHOLD = 0.90
BETA5 = 0.00
BETA3 = 0.01


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sha256_keys(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in df[["trade_date", "stock_code", "pred_prob"]].itertuples(index=False):
        digest.update(f"{row.trade_date}|{row.stock_code}|{float(row.pred_prob):.12g}\n".encode("utf-8"))
    return digest.hexdigest()


def load_feature_slice() -> pd.DataFrame:
    cols = ", ".join(["trade_date", "stock_code", *ftop.FEATURES])
    with duckdb.connect(str(FEATURE_DB), read_only=True) as con:
        return con.execute(
            f"""
            SELECT {cols}
            FROM {FEATURE_TABLE}
            WHERE trade_date >= ?
              AND stock_code NOT LIKE '%.BJ'
            ORDER BY trade_date, stock_code
            """,
            [ftop.rb.START_DATE],
        ).fetchdf()


def prepare_full_base() -> pd.DataFrame:
    base = ftop.rb._add_rank_columns(ftop.rb._read_scores(with_labels=False))
    features = load_feature_slice()
    frame = base.merge(features, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    rank_cols = [f"pred_{key}_rank" for key in ["1d", "3d", "5d", "10d"]]
    for key in ["1d", "3d", "5d", "10d"]:
        others = [col for col in rank_cols if col != f"pred_{key}_rank"]
        frame[f"pred_{key}_gap"] = frame[f"pred_{key}_rank"] - frame[others].mean(axis=1)
    frame["rank_consensus_mean"] = frame[rank_cols].mean(axis=1)
    frame["rank_consensus_min"] = frame[rank_cols].min(axis=1)
    frame["rank_consensus_std"] = frame[rank_cols].std(axis=1)
    for col in ftop.FEATURES:
        frame[f"{col}_rank"] = frame.groupby("trade_date")[col].rank(method="average", pct=True)
    for col in ftop.MODEL_FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.5).astype("float32")
    return frame


def materialize() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = prepare_full_base()
    top5 = xgb.XGBClassifier()
    top5.load_model(str(TOP5_MODEL))
    top3 = xgb.XGBClassifier()
    top3.load_model(str(TOP3_MODEL))
    frame["_p5"] = top5.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p3"] = top3.predict_proba(frame[ftop.MODEL_FEATURES])[:, 1].astype("float32")
    frame["_p5_rank"] = frame.groupby("trade_date")["_p5"].rank(method="average", pct=True)
    frame["_p3_rank"] = frame.groupby("trade_date")["_p3"].rank(method="average", pct=True)

    base_col = "pred_5d_rank"
    mask = frame[base_col] >= POOL_THRESHOLD
    score = frame[base_col].copy()
    score.loc[mask] = (
        frame.loc[mask, base_col]
        + BETA5 * (frame.loc[mask, "_p5_rank"] - 0.5)
        + BETA3 * (frame.loc[mask, "_p3_rank"] - 0.5)
    )

    out = frame[["trade_date", "stock_code"]].copy()
    out["pred_prob"] = score.astype("float64")
    out["score_source"] = "research_5d_front_combo_calibrator_20260714"
    out["model_label"] = "executable_5d_open_return"
    out["created_at"] = now_iso()
    usage = {
        "front_pool_rows": int(mask.sum()),
        "front_pool_row_share": float(mask.mean()),
        "total_rows": int(len(out)),
    }
    return out, usage


def quality(df: pd.DataFrame) -> dict[str, Any]:
    latest = str(df["trade_date"].max())
    latest_frame = df[df["trade_date"] == latest]
    return {
        "row_count": int(len(df)),
        "stock_count": int(df["stock_code"].nunique()),
        "min_trade_date": str(df["trade_date"].min()),
        "max_trade_date": latest,
        "trade_days": int(df["trade_date"].nunique()),
        "latest_day_rows": int(len(latest_frame)),
        "latest_day_stock_count": int(latest_frame["stock_code"].nunique()),
        "bj_rows": int(df["stock_code"].astype(str).str.endswith(".BJ").sum()),
        "null_pred_prob": int(df["pred_prob"].isna().sum()),
        "duplicate_key_groups": int(df.duplicated(["trade_date", "stock_code"]).sum()),
        "pred_prob_sha256": sha256_keys(df.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DB.parent.mkdir(parents=True, exist_ok=True)
    df, usage = materialize()
    q = quality(df)
    with duckdb.connect(str(EXPERIMENT_DB)) as con:
        con.register("candidate_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {quote(TABLE)} AS SELECT * FROM candidate_df")

    manifest = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "actor": "model-agent",
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "governance_status": "research_only_materialized_candidate_pending_audit",
        "source_type": "duckdb_table",
        "db_path": str(EXPERIMENT_DB),
        "table": TABLE,
        "label": "executable_5d_open_return",
        "candidate_id": "research_5d_front_combo_calibrator_20260714",
        "score_formula": (
            "pred_5d_rank + 0.01*(front_top3_rank-0.5), "
            "applied only when pred_5d_rank >= 0.90"
        ),
        "pool_threshold": POOL_THRESHOLD,
        "beta5": BETA5,
        "beta3": BETA3,
        "source_models": {
            "front_top5_model": str(TOP5_MODEL),
            "front_top3_model": str(TOP3_MODEL),
        },
        "source_evaluation_report": str(SOURCE_REPORT),
        "quality": q,
        "score_source_usage": usage,
        "allowed_for_main_workflow": False,
        "boundaries": {
            "research_only": True,
            "not_formal": True,
            "not_approved_for_l5": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    manifest_path = REPORT_DIR / "research_candidate_manifest_5d_front_combo.json"
    report_path = REPORT_DIR / "materialized_research_asset_report_5d_front_combo.json"
    md_path = REPORT_DIR / "materialized_research_asset_report_5d_front_combo.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "scope": "materialize_research_5d_front_combo_calibrator",
        "db_path": str(EXPERIMENT_DB),
        "table": TABLE,
        "quality": q,
        "score_source_usage": usage,
        "manifest": str(manifest_path),
        "source_evaluation_report": str(SOURCE_REPORT),
        "boundaries": manifest["boundaries"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 5D 前排组合校准 Research L4 资产物化报告",
        "",
        "## 结论",
        "",
        "已将 5D 前排组合校准候选物化为 research-only L4 评分资产。该资产不是 formal，不是 approved_for_l5，不能进入生产主线。",
        "",
        "## 资产",
        "",
        f"- DuckDB：`{EXPERIMENT_DB}`",
        f"- 表：`{TABLE}`",
        f"- manifest：`{manifest_path}`",
        "",
        "## 质量",
        "",
        f"- 行数：`{q['row_count']}`",
        f"- 日期范围：`{q['min_trade_date']}` 到 `{q['max_trade_date']}`",
        f"- 交易日：`{q['trade_days']}`",
        f"- 最新日行数 / 股票数：`{q['latest_day_rows']}` / `{q['latest_day_stock_count']}`",
        f"- `.BJ` 行数：`{q['bj_rows']}`",
        f"- `pred_prob` 空值：`{q['null_pred_prob']}`",
        f"- 重复键：`{q['duplicate_key_groups']}`",
        "",
        "## 边界",
        "",
        "- research-only。",
        "- 未修改 formal manifest。",
        "- 未修改 approved_for_l5。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "manifest": str(manifest_path),
                "db": str(EXPERIMENT_DB),
                "table": TABLE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

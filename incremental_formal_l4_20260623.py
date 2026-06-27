from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
QUANT_DIR = ROOT / "quant"
DATA_DIR = QUANT_DIR / "data_file"
MAIN_DIR = QUANT_DIR / "main"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MANIFEST_DIR = MAIN_DIR / "config" / "prediction_manifests"
TARGET_DATE = os.environ.get("MODEL_FORMAL_L4_TARGET_DATE", "20260623")
REPORT_DIR = DATA_DIR / "reports" / f"model_agent_formal_incremental_l4_{TARGET_DATE}"

DB_PATH_IN_MANIFEST = "../../../data_file/model_predictions/MODEL_PREDICTIONS.db"
MARKET_DB_PATH_IN_MANIFEST = "../../../data_file/STOCK_DAILY_DATA.db"

TABLES = {
    "1d_base": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618",
    "5d_gate1": "stock_predict_data_model_agent_5d_tune_20260620_executable_5d_open_return_d4_l4_fs160_gate1_score_20240604_20260618",
    "10d_gate2": "stock_predict_data_model_agent_10d_tune_20260620_executable_10d_open_return_d4_l4_fs160_gate2_score_20240604_20260618",
    "5d_aux": "stock_predict_data_model_agent_5d_aux_horizon_fusion_20260622_executable_5d_open_return_research",
    "5d_topgate": "stock_predict_data_model_agent_5d_topgate_fusion_20260622_executable_5d_open_return_research",
    "5d_gate6_light": "stock_predict_data_model_agent_5d_gate6_top1_light_20260622_executable_5d_open_return_research",
    "5d_balanced": "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research",
    "10d_dense_aux": "stock_predict_data_model_agent_1d_dense_aux_fusion_20260622_executable_1d_open_return_research",
    "10d_topgate": "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research",
    "1d_daygate": "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research",
}

FORMAL_TABLES = {
    "1d": "stock_predict_data_model_agent_best_full_20260623_executable_1d_open_return_formal",
    "3d": "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal",
    "5d": "stock_predict_data_model_agent_best_full_20260623_executable_5d_open_return_formal",
    "10d": "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal",
}

MANIFESTS = {
    "1d": "executable_1d_open_return_l4_formal_20260619.json",
    "3d": "executable_3d_open_return_l4_formal_20260617.json",
    "5d": "executable_5d_open_return_l4_formal_20260620.json",
    "10d": "executable_10d_open_return_l4_formal_20260617.json",
    "prod_5d": "prod_liq_prime_one_v20260612_l4_formal.json",
}

LABELS = {
    "1d": "executable_1d_open_return",
    "3d": "executable_3d_open_return",
    "5d": "executable_5d_open_return",
    "10d": "executable_10d_open_return",
}


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"pragma table_info({quote(table)})")]


def read_day(conn: sqlite3.Connection, table: str, date: str = TARGET_DATE) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select * from {quote(table)} where trade_date = ? order by stock_code",
        conn,
        params=[date],
    )
    if not frame.empty:
        frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def read_score(conn: sqlite3.Connection, table: str, label: str | None = None) -> pd.DataFrame:
    cols = table_columns(conn, table)
    label_select = f", {quote(label)} as label_value" if label and label in cols else ""
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob{label_select} from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    if label and "label_value" not in frame.columns:
        frame["label_value"] = None
    return frame


def add_daily_rank(frame: pd.DataFrame, source_col: str, rank_col: str) -> pd.DataFrame:
    out = frame.copy()
    out[rank_col] = out.groupby("trade_date")[source_col].rank(method="average", pct=True)
    return out


def write_replace_day(conn: sqlite3.Connection, table: str, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    conn.execute(f"delete from {quote(table)} where trade_date = ?", [TARGET_DATE])
    frame.to_sql(table, conn, if_exists="append", index=False)


def write_table(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, label: str) -> None:
    out = frame[["trade_date", "stock_code", "pred_prob"]].copy()
    if "label_value" in frame.columns:
        out[label] = frame["label_value"]
    conn.execute(f"drop table if exists {quote(table)}")
    out.to_sql(table, conn, if_exists="replace", index=False)
    short = hashlib.sha1(table.encode("utf-8")).hexdigest()[:12]
    conn.execute(f"create index if not exists idx_{short}_date_code on {quote(table)}(trade_date, stock_code)")
    conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(table)}(trade_date, pred_prob desc)")


def stats_for_table(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               count(distinct stock_code), sum(case when pred_prob is null then 1 else 0 end)
        from {quote(table)}
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*)
        from (
          select trade_date, stock_code, count(*) as c
          from {quote(table)}
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    latest = conn.execute(
        f"select count(*), count(distinct stock_code) from {quote(table)} where trade_date = ?",
        [TARGET_DATE],
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
        "latest_day_rows": int(latest[0]),
        "latest_day_stock_count": int(latest[1]),
    }


def hash_table(conn: sqlite3.Connection, table: str) -> str:
    digest = hashlib.sha256()
    for trade_date, stock_code, pred_prob in conn.execute(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code"
    ):
        digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.12g}\n".encode("utf-8"))
    return digest.hexdigest()


def extend_10d_topgate(conn: sqlite3.Connection) -> dict:
    base = read_day(conn, TABLES["10d_gate2"])
    if base.empty:
        raise RuntimeError("10D gate2 base table has no 20260623 rows")
    aux = read_day(conn, TABLES["10d_dense_aux"])
    merged = base.rename(columns={"pred_prob": "formal_10d_pred_prob"})
    merged["rank_formal_10d"] = merged["formal_10d_pred_prob"].rank(method="average", pct=True)
    if aux.empty:
        merged["research_1d_dense_pred_prob"] = merged["formal_10d_pred_prob"]
        merged["rank_research_1d_dense"] = merged["rank_formal_10d"]
        fallback = True
    else:
        aux = aux[["trade_date", "stock_code", "pred_prob"]].rename(columns={"pred_prob": "research_1d_dense_pred_prob"})
        merged = merged.merge(aux, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        merged["rank_research_1d_dense"] = merged["research_1d_dense_pred_prob"].rank(method="average", pct=True)
        merged["rank_research_1d_dense"] = merged["rank_research_1d_dense"].fillna(merged["rank_formal_10d"])
        merged["research_1d_dense_pred_prob"] = merged["research_1d_dense_pred_prob"].fillna(merged["formal_10d_pred_prob"])
        fallback = False
    merged["topgate_applied"] = merged["rank_formal_10d"] >= 0.9975
    merged["pred_prob"] = merged["rank_formal_10d"]
    merged.loc[merged["topgate_applied"], "pred_prob"] = (
        merged.loc[merged["topgate_applied"], "rank_formal_10d"]
        + 0.01 * merged.loc[merged["topgate_applied"], "rank_research_1d_dense"]
    )
    keep = [col for col in table_columns(conn, TABLES["10d_topgate"]) if col in merged.columns]
    write_replace_day(conn, TABLES["10d_topgate"], merged[keep])
    return {
        "table": TABLES["10d_topgate"],
        "rows": int(len(merged)),
        "topgate_rows": int(merged["topgate_applied"].sum()),
        "aux_fallback_to_base": fallback,
    }


def extend_5d_topgate(conn: sqlite3.Connection) -> dict:
    base = read_day(conn, TABLES["5d_aux"])
    base_source = TABLES["5d_aux"]
    if base.empty:
        base = read_day(conn, TABLES["5d_gate1"])
        base_source = TABLES["5d_gate1"]
    formal = read_day(conn, TABLES["5d_gate1"])
    if base.empty or formal.empty:
        raise RuntimeError("5D topgate base/formal table has no 20260623 rows")
    merged = base.rename(columns={"pred_prob": "base_5d_aux_pred_prob"})
    formal = formal[["trade_date", "stock_code", "pred_prob"]].rename(columns={"pred_prob": "formal_5d_pred_prob"})
    if "formal_5d_pred_prob" not in merged.columns:
        merged = merged.merge(formal, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    merged["rank_base_5d_aux"] = merged["base_5d_aux_pred_prob"].rank(method="average", pct=True)
    merged["rank_formal_5d"] = merged["formal_5d_pred_prob"].rank(method="average", pct=True)
    merged["rank_formal_5d"] = merged["rank_formal_5d"].fillna(merged["rank_base_5d_aux"])
    merged["topgate_applied"] = merged["rank_base_5d_aux"] >= 0.9975
    merged["pred_prob"] = merged["rank_base_5d_aux"]
    merged.loc[merged["topgate_applied"], "pred_prob"] = (
        merged.loc[merged["topgate_applied"], "rank_base_5d_aux"]
        + 0.001 * merged.loc[merged["topgate_applied"], "rank_formal_5d"]
    )
    keep = [col for col in table_columns(conn, TABLES["5d_topgate"]) if col in merged.columns]
    write_replace_day(conn, TABLES["5d_topgate"], merged[keep])
    return {
        "table": TABLES["5d_topgate"],
        "rows": int(len(merged)),
        "topgate_rows": int(merged["topgate_applied"].sum()),
        "base_source": base_source,
    }


def extend_5d_balanced(conn: sqlite3.Connection) -> dict:
    base = read_day(conn, TABLES["5d_topgate"])
    if base.empty:
        raise RuntimeError("5D topgate table has no 20260623 rows")
    gate6 = read_day(conn, TABLES["5d_gate6_light"])
    merged = base[["trade_date", "stock_code", "pred_prob"]].rename(columns={"pred_prob": "base_pred_prob"})
    label_col = LABELS["5d"]
    if label_col in base.columns:
        merged[label_col] = base[label_col]
    merged["formal_score"] = merged["base_pred_prob"]
    merged["base_rank"] = merged["base_pred_prob"].rank(method="average", pct=True)
    if gate6.empty:
        merged["gate6_pred_prob"] = None
        merged["gate6_rank"] = merged["base_rank"]
        merged["bonus_applied"] = False
        merged["score_source"] = "base_fallback_no_gate6_20260623"
        merged["pred_prob"] = merged["base_rank"]
        fallback = True
    else:
        gate6 = gate6[["trade_date", "stock_code", "pred_prob"]].rename(columns={"pred_prob": "gate6_pred_prob"})
        merged = merged.merge(gate6, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        merged["gate6_rank"] = merged["gate6_pred_prob"].rank(method="average", pct=True)
        merged["gate6_rank"] = merged["gate6_rank"].fillna(merged["base_rank"])
        merged["bonus_applied"] = merged["base_rank"] >= 0.999
        merged["pred_prob"] = merged["base_rank"]
        merged.loc[merged["bonus_applied"], "pred_prob"] = (
            merged.loc[merged["bonus_applied"], "base_rank"] + 0.001 * merged.loc[merged["bonus_applied"], "gate6_rank"]
        )
        merged["score_source"] = "gate6_bonus"
        fallback = False
    keep = [col for col in table_columns(conn, TABLES["5d_balanced"]) if col in merged.columns]
    write_replace_day(conn, TABLES["5d_balanced"], merged[keep])
    return {
        "table": TABLES["5d_balanced"],
        "rows": int(len(merged)),
        "bonus_rows": int(merged["bonus_applied"].sum()),
        "gate6_fallback_to_base": fallback,
    }


def build_3d(conn: sqlite3.Connection) -> dict:
    ten = add_daily_rank(read_score(conn, TABLES["10d_topgate"], LABELS["3d"]), "pred_prob", "rank10")
    five = add_daily_rank(read_score(conn, TABLES["5d_balanced"]), "pred_prob", "rank5")
    frame = ten[["trade_date", "stock_code", "rank10", "label_value"]].merge(
        five[["trade_date", "stock_code", "rank5"]],
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    frame["pred_prob"] = 0.88 * frame["rank10"] + 0.12 * frame["rank5"]
    write_table(conn, FORMAL_TABLES["3d"], frame, LABELS["3d"])
    return {
        "formula": "score = 0.88 * daily_rank(current_full_10d) + 0.12 * daily_rank(current_full_5d)",
        "sources": [TABLES["10d_topgate"], TABLES["5d_balanced"]],
    }


def build_5d(conn: sqlite3.Connection) -> dict:
    base = add_daily_rank(read_score(conn, TABLES["5d_balanced"], LABELS["5d"]), "pred_prob", "rank_base")
    aux = add_daily_rank(read_score(conn, FORMAL_TABLES["3d"]), "pred_prob", "rank3")
    frame = base[["trade_date", "stock_code", "rank_base", "label_value"]].merge(
        aux[["trade_date", "stock_code", "rank3"]],
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    mask = (frame["trade_date"] >= "20260301") & (frame["rank_base"] >= 0.994)
    frame["pred_prob"] = frame["rank_base"]
    frame.loc[mask, "pred_prob"] = frame.loc[mask, "rank_base"] + 0.003 * frame.loc[mask, "rank3"]
    write_table(conn, FORMAL_TABLES["5d"], frame, LABELS["5d"])
    return {
        "formula": "base = daily_rank(current_full_5d); if trade_date >= 20260301 and base >= 0.994: score = base + 0.003 * daily_rank(full_3d_best); else score = base",
        "sources": [TABLES["5d_balanced"], FORMAL_TABLES["3d"]],
        "active_rows": int(mask.sum()),
        "active_trade_days": int(frame.loc[mask, "trade_date"].nunique()),
    }


def build_10d(conn: sqlite3.Connection) -> dict:
    base = add_daily_rank(read_score(conn, TABLES["10d_topgate"], LABELS["10d"]), "pred_prob", "rank_base")
    proxy = add_daily_rank(read_score(conn, FORMAL_TABLES["5d"]), "pred_prob", "rank_proxy")
    frame = base[["trade_date", "stock_code", "rank_base", "label_value"]].merge(
        proxy[["trade_date", "stock_code", "rank_proxy"]],
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    day_gate = frame.groupby("trade_date", sort=True).apply(
        lambda g: float(g.nlargest(5, "rank_base")["rank_base"].std() or 0.0),
        include_groups=False,
    ).reset_index(name="std_top5_base")
    frame = frame.merge(day_gate, on="trade_date", how="left")
    mask = (frame["trade_date"] >= "20260101") & (frame["std_top5_base"] >= 0.05) & (frame["std_top5_base"] < 0.35)
    frame["pred_prob"] = frame["rank_base"]
    frame.loc[mask, "pred_prob"] = frame.loc[mask, "rank_proxy"]
    write_table(conn, FORMAL_TABLES["10d"], frame, LABELS["10d"])
    day_gate.to_csv(REPORT_DIR / "10d_full_coverage_day_gate.csv", index=False, encoding="utf-8-sig")
    return {
        "formula": "base = daily_rank(current_full_10d); proxy = daily_rank(full_5d_best); if trade_date >= 20260101 and 0.05 <= std(base top5) < 0.35: score = proxy; else score = base",
        "sources": [TABLES["10d_topgate"], FORMAL_TABLES["5d"]],
        "switch_rows": int(mask.sum()),
        "switch_trade_days": int(frame.loc[mask, "trade_date"].nunique()),
        "day_gate_csv": str((REPORT_DIR / "10d_full_coverage_day_gate.csv").as_posix()),
    }


def build_1d(conn: sqlite3.Connection) -> dict:
    one = add_daily_rank(read_score(conn, TABLES["1d_base"], LABELS["1d"]), "pred_prob", "rank1")
    ten = add_daily_rank(read_score(conn, FORMAL_TABLES["10d"]), "pred_prob", "rank10")
    day = add_daily_rank(read_score(conn, TABLES["1d_daygate"]), "pred_prob", "rank_daygate")
    frame = one[["trade_date", "stock_code", "rank1", "label_value"]].merge(
        ten[["trade_date", "stock_code", "rank10"]],
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    ).merge(
        day[["trade_date", "stock_code", "rank_daygate"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="one_to_one",
    )
    parts = []
    choice_rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        ordered_one = group.sort_values("rank1", ascending=False, kind="mergesort").reset_index(drop=True)
        current_gap = float(ordered_one.iloc[0]["rank1"] - ordered_one.iloc[4]["rank1"])
        use_one = current_gap > 0.000795
        selected = group[["trade_date", "stock_code", "rank1", "rank10", "rank_daygate", "label_value"]].copy()
        selected["default_score"] = selected["rank1"] if use_one else selected["rank10"]
        ordered_default = selected.sort_values("default_score", ascending=False, kind="mergesort").reset_index(drop=True)
        best_gap = float(ordered_default.iloc[0]["default_score"] - ordered_default.iloc[4]["default_score"])
        use_daygate = trade_date <= "20251231" and best_gap <= 0.000795
        selected["pred_prob"] = selected["default_score"]
        if use_daygate:
            selected["pred_prob"] = selected["rank_daygate"].where(selected["rank_daygate"].notna(), selected["default_score"])
        parts.append(selected)
        choice_rows.append(
            {
                "trade_date": trade_date,
                "default_source": "current_full_1d" if use_one else "full_10d_best",
                "current_gap": current_gap,
                "best_gap": best_gap,
                "use_daygate": bool(use_daygate),
            }
        )
    output = pd.concat(parts, ignore_index=True)
    write_table(conn, FORMAL_TABLES["1d"], output, LABELS["1d"])
    choices = pd.DataFrame(choice_rows)
    choices.to_csv(REPORT_DIR / "1d_full_coverage_daily_choices.csv", index=False, encoding="utf-8-sig")
    return {
        "formula": "default = current_full_1d when current top1-top5 rank gap > 0.000795 else full_10d_best; if trade_date <= 20251231 and default top1-top5 gap <= 0.000795: use full 1d daygate score; else default",
        "sources": [TABLES["1d_base"], FORMAL_TABLES["10d"], TABLES["1d_daygate"]],
        "daily_choices_csv": str((REPORT_DIR / "1d_full_coverage_daily_choices.csv").as_posix()),
        "daygate_days": int(choices["use_daygate"].sum()),
    }


def archive_manifest(name: str) -> str:
    src = MANIFEST_DIR / name
    archive = src.with_name(src.stem + f"_archive_before_{TARGET_DATE}_incremental_l4.json")
    if not archive.exists():
        shutil.copy2(src, archive)
    return archive.name


def update_manifest(name: str, label_key: str, stats: dict, formula_info: dict, archive_name: str) -> None:
    path = MANIFEST_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "5d" if label_key == "prod_5d" else label_key
    payload.update(
        {
            "schema_version": payload.get("schema_version", 1),
            "asset_role": "l4_formal_prediction_asset",
            "approval_status": "approved_for_l5",
            "source_type": "sqlite_table",
            "db_path": DB_PATH_IN_MANIFEST,
            "table": FORMAL_TABLES[key],
            "market_db_path": MARKET_DB_PATH_IN_MANIFEST,
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
                "trade_date": TARGET_DATE,
                "rows": stats["latest_day_rows"],
                "method": "saved_production_model_and_approved_formula_incremental_l4_scoring",
                "production_factor_input": "quant/data_file/production_factor_parts/",
                "no_training": True,
                "no_new_research_model": True,
                "no_signal": True,
                "no_backtest": True,
                "report_path": "../../../data_file/reports/model_agent_formal_incremental_l4_20260623/formal_incremental_l4_20260623_report.json",
            },
            "previous_manifest_archive_before_incremental_l4_20260623": archive_name,
            "notes": "使用生产因子数据补齐 20260623 生产预测结果；仅执行已保存生产模型推理和已发布评分公式增量计算，未训练、未调参、未生成交易信号、未运行回测。",
            "rollback_note": "如需回滚，恢复 previous_manifest_archive_before_incremental_l4_20260623 指向的 manifest；本次未删除旧表。",
        }
    )
    if label_key == "prod_5d":
        payload["strategy_id"] = "prod_liq_prime_one_v20260612"
        payload["governance_status"] = "approved_l5_consumable_strategy_specific_compat_manifest_not_default_task"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_manifest_dynamic(name: str, label_key: str, stats: dict, formula_info: dict, archive_name: str) -> None:
    path = MANIFEST_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "5d" if label_key == "prod_5d" else label_key
    payload.update(
        {
            "schema_version": payload.get("schema_version", 1),
            "asset_role": "l4_formal_prediction_asset",
            "approval_status": "approved_for_l5",
            "source_type": "sqlite_table",
            "db_path": DB_PATH_IN_MANIFEST,
            "table": FORMAL_TABLES[key],
            "market_db_path": MARKET_DB_PATH_IN_MANIFEST,
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
                "trade_date": TARGET_DATE,
                "rows": stats["latest_day_rows"],
                "method": "saved_production_model_and_approved_formula_incremental_l4_scoring",
                "production_factor_input": "quant/data_file/production_factor_parts/",
                "no_training": True,
                "no_new_research_model": True,
                "no_signal": True,
                "no_backtest": True,
                "report_path": f"../../../data_file/reports/model_agent_formal_incremental_l4_{TARGET_DATE}/formal_incremental_l4_{TARGET_DATE}_report.json",
            },
            f"previous_manifest_archive_before_incremental_l4_{TARGET_DATE}": archive_name,
            "notes": f"使用生产因子数据补齐 {TARGET_DATE} 生产预测结果；仅执行已保存生产模型推理和已发布评分公式增量计算，未训练、未调参、未生成交易信号、未运行回测。",
            "rollback_note": f"如需回滚，恢复 previous_manifest_archive_before_incremental_l4_{TARGET_DATE} 指向的 manifest；本次未删除旧表。",
        }
    )
    if label_key == "prod_5d":
        payload["strategy_id"] = "prod_liq_prime_one_v20260612"
        payload["governance_status"] = "approved_l5_consumable_strategy_specific_compat_manifest_not_default_task"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_manifest(path: Path) -> dict:
    import sys

    sys.path.insert(0, str(MAIN_DIR))
    from prediction_manifest import load_prediction_source_manifest

    source = load_prediction_source_manifest(str(path), require_approved=True, allow_legacy=False)
    if isinstance(source, dict):
        return {
            "manifest": str(path.as_posix()),
            "db_path": str(source.get("db_path")),
            "table": str(source.get("table") or source.get("prediction_table")),
        }
    return {"manifest": str(path.as_posix()), "db_path": str(source.db_path), "table": str(source.table)}


def write_report(report: dict) -> None:
    (REPORT_DIR / "formal_incremental_l4_20260623_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 20260623 生产预测增量补齐报告",
        "",
        "## 当前结论",
        "",
        "已按生产因子数据补齐 20260623 的 L4 formal 生产预测结果。补齐对象为 1D、3D、5D、10D 四个当前 formal 表，以及 `prod_liq_prime_one` 兼容 5D manifest。",
        "",
        "## 执行边界",
        "",
        "- 未训练模型",
        "- 未调参",
        "- 未生成交易信号",
        "- 未运行回测",
        "- 未修改策略规则",
        "",
        "## 表覆盖",
        "",
    ]
    for key, table in FORMAL_TABLES.items():
        s = report["formal_stats"][key]
        lines.extend(
            [
                f"- `{key}`：`{table}`",
                f"  - 日期：`{s['min_trade_date']}` 到 `{s['max_trade_date']}`",
                f"  - 总行数：`{s['row_count']}`，最新日行数：`{s['latest_day_rows']}`",
                f"  - 空分数：`{s['null_pred_prob']}`，重复键：`{s['duplicate_key_groups']}`",
            ]
        )
    lines.extend(["", "## 组件补齐", ""])
    for item in report["component_updates"]:
        lines.append(f"- `{item['table']}`：`{item['rows']}` 行，详情：`{json.dumps(item, ensure_ascii=False)}`")
    lines.extend(["", "## 证据", "", f"- JSON 报告：`{(REPORT_DIR / 'formal_incremental_l4_20260623_report.json').as_posix()}`"])
    (REPORT_DIR / "formal_incremental_l4_20260623_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_report_dynamic(report: dict) -> None:
    report_json = REPORT_DIR / f"formal_incremental_l4_{TARGET_DATE}_report.json"
    report_md = REPORT_DIR / f"formal_incremental_l4_{TARGET_DATE}_report.md"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# {TARGET_DATE} 生产预测增量补齐报告",
        "",
        "## 当前结论",
        "",
        f"已按生产因子数据补齐 {TARGET_DATE} 的 L4 formal 生产预测结果。补齐对象为 1D、3D、5D、10D 四个当前 formal 表，以及 `prod_liq_prime_one` 兼容 5D manifest。",
        "",
        "## 执行边界",
        "",
        "- 未训练模型",
        "- 未调参",
        "- 未生成交易信号",
        "- 未运行回测",
        "- 未修改策略规则",
        "",
        "## 表覆盖",
        "",
    ]
    for key, table in FORMAL_TABLES.items():
        s = report["formal_stats"][key]
        lines.extend(
            [
                f"- `{key}`：`{table}`",
                f"  - 日期：`{s['min_trade_date']}` 到 `{s['max_trade_date']}`",
                f"  - 总行数：`{s['row_count']}`，最新日行数：`{s['latest_day_rows']}`",
                f"  - 空分数：`{s['null_pred_prob']}`，重复键：`{s['duplicate_key_groups']}`",
            ]
        )
    lines.extend(["", "## 组件补齐", ""])
    for item in report["component_updates"]:
        lines.append(f"- `{item['table']}`：`{item['rows']}` 行，详情：`{json.dumps(item, ensure_ascii=False)}`")
    lines.extend(["", "## 证据", "", f"- JSON 报告：`{report_json.as_posix()}`"])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    formula_info: dict[str, dict] = {}
    stats: dict[str, dict] = {}
    component_updates: list[dict] = []

    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        component_updates.append(extend_10d_topgate(conn))
        component_updates.append(extend_5d_topgate(conn))
        component_updates.append(extend_5d_balanced(conn))
        formula_info["3d"] = build_3d(conn)
        formula_info["5d"] = build_5d(conn)
        formula_info["10d"] = build_10d(conn)
        formula_info["1d"] = build_1d(conn)
        conn.commit()
        for key, table in FORMAL_TABLES.items():
            table_stats = stats_for_table(conn, table)
            table_stats["pred_prob_sha256"] = hash_table(conn, table)
            stats[key] = table_stats

    archives = {key: archive_manifest(name) for key, name in MANIFESTS.items()}
    update_manifest_dynamic(MANIFESTS["1d"], "1d", stats["1d"], formula_info["1d"], archives["1d"])
    update_manifest_dynamic(MANIFESTS["3d"], "3d", stats["3d"], formula_info["3d"], archives["3d"])
    update_manifest_dynamic(MANIFESTS["5d"], "5d", stats["5d"], formula_info["5d"], archives["5d"])
    update_manifest_dynamic(MANIFESTS["10d"], "10d", stats["10d"], formula_info["10d"], archives["10d"])
    update_manifest_dynamic(MANIFESTS["prod_5d"], "prod_5d", stats["5d"], formula_info["5d"], archives["prod_5d"])

    validation = [validate_manifest(MANIFEST_DIR / name) for name in MANIFESTS.values()]
    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "target_date": TARGET_DATE,
        "db_path": str(MODEL_DB),
        "formal_tables": FORMAL_TABLES,
        "formal_stats": stats,
        "component_updates": component_updates,
        "formula_info": formula_info,
        "manifest_archives": archives,
        "manifest_validation": validation,
        "boundaries": {
            "no_training": True,
            "no_tuning": True,
            "no_signal": True,
            "no_backtest": True,
            "no_strategy_rule_change": True,
        },
    }
    write_report_dynamic(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

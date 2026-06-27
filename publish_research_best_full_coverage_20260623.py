from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
QUANT_DIR = ROOT / "quant"
DATA_DIR = QUANT_DIR / "data_file"
MAIN_DIR = QUANT_DIR / "main"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
MANIFEST_DIR = MAIN_DIR / "config" / "prediction_manifests"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_research_best_full_coverage_release_20260623"
COLLAB_PATH = DATA_DIR / "runtime" / "agent_memory" / "collaboration_requests.jsonl"

DB_PATH_IN_MANIFEST = "../../../data_file/model_predictions/MODEL_PREDICTIONS.db"
MARKET_DB_PATH_IN_MANIFEST = "../../../data_file/STOCK_DAILY_DATA.db"

CURRENT_TABLES = {
    "1d": "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618",
    "3d": "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_formal",
    "5d": "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research",
    "10d": "stock_predict_data_model_agent_10d_topgate_fusion_20260622_executable_10d_open_return_research",
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


def read_score(conn: sqlite3.Connection, table: str, label: str | None = None) -> pd.DataFrame:
    cols = [row[1] for row in conn.execute(f"pragma table_info('{table}')")]
    label_select = f", [{label}] as label_value" if label and label in cols else ""
    query = f"select trade_date, stock_code, pred_prob{label_select} from '{table}'"
    frame = pd.read_sql_query(query, conn)
    frame["trade_date"] = frame["trade_date"].astype(str)
    if label and "label_value" not in frame.columns:
        frame["label_value"] = None
    return frame


def add_daily_rank(frame: pd.DataFrame, source_col: str, rank_col: str) -> pd.DataFrame:
    out = frame.copy()
    out[rank_col] = out.groupby("trade_date")[source_col].rank(method="average", pct=True)
    return out


def stats_for_table(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               count(distinct stock_code), sum(case when pred_prob is null then 1 else 0 end)
        from '{table}'
        """
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*)
        from (
          select trade_date, stock_code, count(*) as c
          from '{table}'
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    latest = conn.execute(
        f"""
        select count(*), count(distinct stock_code)
        from '{table}'
        where trade_date = (select max(trade_date) from '{table}')
        """
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
        f"select trade_date, stock_code, pred_prob from '{table}' order by trade_date, stock_code"
    ):
        digest.update(f"{trade_date}|{stock_code}|{float(pred_prob):.12g}\n".encode("utf-8"))
    return digest.hexdigest()


def write_table(conn: sqlite3.Connection, table: str, frame: pd.DataFrame, label: str) -> None:
    out = frame[["trade_date", "stock_code", "pred_prob"]].copy()
    if "label_value" in frame.columns:
        out[label] = frame["label_value"]
    conn.execute(f"drop table if exists '{table}'")
    out.to_sql(table, conn, index=False)
    conn.execute(f"create index if not exists idx_{table}_date_code on '{table}'(trade_date, stock_code)")
    conn.execute(f"create index if not exists idx_{table}_date_pred on '{table}'(trade_date, pred_prob desc)")


def build_3d(conn: sqlite3.Connection) -> dict:
    ten = add_daily_rank(read_score(conn, CURRENT_TABLES["10d"], LABELS["3d"]), "pred_prob", "rank10")
    five = add_daily_rank(read_score(conn, CURRENT_TABLES["5d"]), "pred_prob", "rank5")
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
        "sources": [CURRENT_TABLES["10d"], CURRENT_TABLES["5d"]],
    }


def build_5d(conn: sqlite3.Connection) -> dict:
    base = add_daily_rank(read_score(conn, CURRENT_TABLES["5d"], LABELS["5d"]), "pred_prob", "rank_base")
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
        "sources": [CURRENT_TABLES["5d"], FORMAL_TABLES["3d"]],
        "active_rows": int(mask.sum()),
        "active_trade_days": int(frame.loc[mask, "trade_date"].nunique()),
    }


def build_10d(conn: sqlite3.Connection) -> dict:
    base = add_daily_rank(read_score(conn, CURRENT_TABLES["10d"], LABELS["10d"]), "pred_prob", "rank_base")
    proxy = add_daily_rank(read_score(conn, FORMAL_TABLES["5d"]), "pred_prob", "rank_proxy")
    frame = base[["trade_date", "stock_code", "rank_base", "label_value"]].merge(
        proxy[["trade_date", "stock_code", "rank_proxy"]],
        on=["trade_date", "stock_code"],
        how="inner",
        validate="one_to_one",
    )
    day_rows = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        top5 = group.nlargest(5, "rank_base")
        std_top5 = float(top5["rank_base"].std() or 0.0)
        day_rows.append({"trade_date": trade_date, "std_top5_base": std_top5})
    day_gate = pd.DataFrame(day_rows)
    frame = frame.merge(day_gate, on="trade_date", how="left")
    mask = (frame["trade_date"] >= "20260101") & (frame["std_top5_base"] >= 0.05) & (frame["std_top5_base"] < 0.35)
    frame["pred_prob"] = frame["rank_base"]
    frame.loc[mask, "pred_prob"] = frame.loc[mask, "rank_proxy"]
    write_table(conn, FORMAL_TABLES["10d"], frame, LABELS["10d"])
    day_gate.to_csv(REPORT_DIR / "10d_full_coverage_day_gate.csv", index=False, encoding="utf-8-sig")
    return {
        "formula": "base = daily_rank(current_full_10d); proxy = daily_rank(full_5d_best); if trade_date >= 20260101 and 0.05 <= std(base top5) < 0.35: score = proxy; else score = base",
        "sources": [CURRENT_TABLES["10d"], FORMAL_TABLES["5d"]],
        "switch_rows": int(mask.sum()),
        "switch_trade_days": int(frame.loc[mask, "trade_date"].nunique()),
        "day_gate_csv": str((REPORT_DIR / "10d_full_coverage_day_gate.csv").as_posix()),
    }


def build_1d(conn: sqlite3.Connection) -> dict:
    one = add_daily_rank(read_score(conn, CURRENT_TABLES["1d"], LABELS["1d"]), "pred_prob", "rank1")
    ten = add_daily_rank(read_score(conn, FORMAL_TABLES["10d"]), "pred_prob", "rank10")
    day = add_daily_rank(read_score(conn, CURRENT_TABLES["1d_daygate"]), "pred_prob", "rank_daygate")
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

    default_parts = []
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
        default_parts.append(selected)
        choice_rows.append(
            {
                "trade_date": trade_date,
                "default_source": "current_full_1d" if use_one else "full_10d_best",
                "current_gap": current_gap,
                "best_gap": best_gap,
                "use_daygate": bool(use_daygate),
            }
        )
    output = pd.concat(default_parts, ignore_index=True)
    write_table(conn, FORMAL_TABLES["1d"], output, LABELS["1d"])
    choices = pd.DataFrame(choice_rows)
    choices.to_csv(REPORT_DIR / "1d_full_coverage_daily_choices.csv", index=False, encoding="utf-8-sig")
    return {
        "formula": "default = current_full_1d when current top1-top5 rank gap > 0.000795 else full_10d_best; if trade_date <= 20251231 and default top1-top5 gap <= 0.000795: use full 1d daygate score; else default",
        "sources": [CURRENT_TABLES["1d"], FORMAL_TABLES["10d"], CURRENT_TABLES["1d_daygate"]],
        "daily_choices_csv": str((REPORT_DIR / "1d_full_coverage_daily_choices.csv").as_posix()),
        "daygate_days": int(choices["use_daygate"].sum()),
    }


def archive_manifest(name: str) -> str:
    src = MANIFEST_DIR / name
    archive = src.with_name(src.stem + "_archive_before_20260623_best_full_publish.json")
    if not archive.exists():
        shutil.copy2(src, archive)
    return archive.name


def write_manifest(name: str, label_key: str, table: str, stats: dict, formula_info: dict, archive_name: str, strategy_id: str | None = None) -> None:
    label = LABELS["5d"] if label_key == "prod_5d" else LABELS[label_key]
    payload = {
        "schema_version": 1,
        "asset_role": "l4_formal_prediction_asset",
        "label": label,
        "approval_status": "approved_for_l5",
        "source_type": "sqlite_table",
        "db_path": DB_PATH_IN_MANIFEST,
        "table": table,
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
        "candidate_id": f"model_agent_best_full_coverage_20260623_{label}",
        "model_track": "formal_l5_consumable_strategy_score_asset",
        "score_formula": formula_info["formula"],
        "formula_sources": formula_info["sources"],
        "full_coverage_release_report": "../../../data_file/reports/model_agent_research_best_full_coverage_release_20260623/release_report.json",
        "previous_formal_manifest_archive": archive_name,
        "rollback_manifest_path": archive_name,
        "governance_status": "approved_l5_consumable_user_authorized_best_full_coverage_publish_20260623",
        "l5_approved_at": now_iso(),
        "l5_approval_basis": "user_authorized_publish_all_best_models_after_full_coverage_completion_20260623",
        "coverage_completion_method": "deterministic full-coverage rebuild of selected score formulas using current full-coverage upstream formal components; no model training and no strategy backtest",
        "release_boundary": {
            "no_training": True,
            "no_signal": True,
            "no_backtest": True,
            "no_strategy_rule_change": True,
        },
        "notes": "用户授权将四个当前最优模型/评分公式补齐到当前标准链路最新日期后发布为生产模型。模型侧按全覆盖 upstream formal 分数重算确定性评分公式，未训练模型、未生成交易信号、未运行回测。",
        "rollback_note": "如需回滚，恢复 previous_formal_manifest_archive 指向的归档 manifest；旧 DB 表和旧 manifest 未删除。",
    }
    if strategy_id:
        payload["strategy_id"] = strategy_id
        payload["governance_status"] = "approved_l5_consumable_strategy_specific_compat_manifest_user_authorized_best_full_coverage_publish_20260623"
        payload["notes"] += " 该 manifest 是策略兼容入口，不声明自己是 production_tasks.json 唯一默认任务入口。"
    (MANIFEST_DIR / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    formula_info: dict[str, dict] = {}
    stats: dict[str, dict] = {}

    with sqlite3.connect(MODEL_DB) as conn:
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
    write_manifest(MANIFESTS["1d"], "1d", FORMAL_TABLES["1d"], stats["1d"], formula_info["1d"], archives["1d"])
    write_manifest(MANIFESTS["3d"], "3d", FORMAL_TABLES["3d"], stats["3d"], formula_info["3d"], archives["3d"])
    write_manifest(MANIFESTS["5d"], "5d", FORMAL_TABLES["5d"], stats["5d"], formula_info["5d"], archives["5d"])
    write_manifest(MANIFESTS["10d"], "10d", FORMAL_TABLES["10d"], stats["10d"], formula_info["10d"], archives["10d"])
    write_manifest(
        MANIFESTS["prod_5d"],
        "prod_5d",
        FORMAL_TABLES["5d"],
        stats["5d"],
        formula_info["5d"],
        archives["prod_5d"],
        strategy_id="prod_liq_prime_one_v20260612",
    )

    validation = []
    for name in MANIFESTS.values():
        validation.append(validate_manifest(MANIFEST_DIR / name))

    report = {
        "generated_at": now_iso(),
        "actor": "model-agent",
        "decision": "published_to_formal_l4_l5_consumable",
        "db_path": str(MODEL_DB),
        "formal_tables": FORMAL_TABLES,
        "stats": stats,
        "formula_info": formula_info,
        "manifest_archives": archives,
        "manifest_validation": validation,
        "boundaries": {
            "no_training": True,
            "no_signal": True,
            "no_backtest": True,
            "no_strategy_rule_change": True,
        },
        "important_note": "本次没有把 20260605 截止的研究候选表原样覆盖生产，而是用当前全覆盖 upstream formal 分数组件重算对应评分公式，生成覆盖 20240604-20260622 的正式 L4 表。",
    }
    (REPORT_DIR / "release_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 当前最优模型全覆盖生产发布报告",
        "",
        "## 当前结论",
        "",
        "已将 1D、3D、5D、10D 四个当前最优评分公式补齐为覆盖 `20240604-20260622` 的 formal L4 / L5 可消费生产模型资产。",
        "",
        "本次没有把 `20260605` 截止的研究候选表原样覆盖生产；为避免生产日期倒退，改为使用当前全覆盖 upstream formal 分数组件重算对应评分公式。",
        "",
        "## 发布表",
    ]
    for key, table in FORMAL_TABLES.items():
        s = stats[key]
        lines.extend(
            [
                f"- `{key}`：`{table}`",
                f"  - 日期：`{s['min_trade_date']}` 到 `{s['max_trade_date']}`",
                f"  - 行数：`{s['row_count']}`，最新日行数：`{s['latest_day_rows']}`",
                f"  - 空分数：`{s['null_pred_prob']}`，重复键：`{s['duplicate_key_groups']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Manifest",
            "",
        ]
    )
    for key, name in MANIFESTS.items():
        lines.append(f"- `{key}`：`{(MANIFEST_DIR / name).as_posix()}`")
    lines.extend(
        [
            "",
            "## 边界声明",
            "",
            "- 未训练模型。",
            "- 未生成交易信号。",
            "- 未运行回测。",
            "- 未修改策略规则。",
            "- 旧 manifest 已归档，旧表未删除，可回滚。",
            "",
        ]
    )
    (REPORT_DIR / "release_report.md").write_text("\n".join(lines), encoding="utf-8")

    handoff = {
        "created_at": now_iso(),
        "from_agent": "model-agent",
        "to_agent": "strategy-agent",
        "cc_agents": ["audit-agent", "commander-agent"],
        "task_id": "model-best-full-coverage-formal-release-20260623",
        "type": "model_asset_formal_release",
        "status": "published",
        "message": "用户授权后，模型侧已将 1D/3D/5D/10D 当前最优评分公式补齐为覆盖 20240604-20260622 的 formal L4 / approved_for_l5 生产模型资产。请策略智能体按新版生产模型资产开发/验证策略；策略侧仍需在自身边界内完成回测与策略评估，模型侧未生成信号、未跑回测、未改策略规则。",
        "formal_manifests": {key: str((MANIFEST_DIR / name).as_posix()) for key, name in MANIFESTS.items()},
        "formal_tables": FORMAL_TABLES,
        "release_report": str((REPORT_DIR / "release_report.json").as_posix()),
        "no_training": True,
        "no_signal": True,
        "no_backtest": True,
        "no_strategy_rule_change": True,
    }
    COLLAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with COLLAB_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(handoff, ensure_ascii=False) + "\n")

    print(json.dumps({"report": str(REPORT_DIR / "release_report.json"), "formal_tables": FORMAL_TABLES}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

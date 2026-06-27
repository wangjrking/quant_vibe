from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
OUT_DIR = DATA_DIR / "reports" / "model_agent_extend_10d_no_star_research_20260624"

SOURCE_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_20260623_executable_10d_open_return_research"
FORMAL_TABLE = "stock_predict_data_model_agent_best_full_20260623_executable_10d_open_return_formal"
TARGET_TABLE = "stock_predict_data_model_agent_10d_no_star_r3gate991_r5gate98_ext_20260624_executable_10d_open_return_research"
LABEL_COL = "executable_10d_open_return"
TARGET_DATE = "20260623"


def _read_frame(conn: sqlite3.Connection, table: str, where: str = "") -> pd.DataFrame:
    frame = pd.read_sql_query(f"select * from '{table}' {where}", conn)
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "stock_code" in frame.columns:
        frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def _table_summary(conn: sqlite3.Connection, table: str) -> dict:
    row = conn.execute(
        f"""
        select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
               count(distinct stock_code),
               sum(case when trade_date = ? then 1 else 0 end),
               sum(case when pred_prob is null then 1 else 0 end)
        from '{table}'
        """,
        (TARGET_DATE,),
    ).fetchone()
    dup = conn.execute(
        f"""
        select count(*) from (
          select trade_date, stock_code, count(*) c
          from '{table}'
          group by trade_date, stock_code
          having c > 1
        )
        """
    ).fetchone()[0]
    return {
        "table": table,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "target_date_rows": int(row[5] or 0),
        "null_pred_prob": int(row[6] or 0),
        "duplicate_key_groups": int(dup),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    with sqlite3.connect(MODEL_DB) as conn:
        source = _read_frame(conn, SOURCE_TABLE, f"where trade_date < '{TARGET_DATE}'")
        latest = _read_frame(conn, FORMAL_TABLE, f"where trade_date = '{TARGET_DATE}'")

        if latest.empty:
            raise RuntimeError(f"{FORMAL_TABLE} has no {TARGET_DATE} rows")

        needed = ["trade_date", "stock_code", "pred_prob"]
        if LABEL_COL in source.columns and LABEL_COL in latest.columns:
            needed.append(LABEL_COL)
        output = pd.concat([source[needed], latest[needed]], ignore_index=True)

        conn.execute(f"drop table if exists '{TARGET_TABLE}'")
        output.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{TARGET_TABLE}_date_code on '{TARGET_TABLE}'(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{TARGET_TABLE}_date_pred on '{TARGET_TABLE}'(trade_date, pred_prob desc)")

        source_summary = _table_summary(conn, SOURCE_TABLE)
        formal_summary = _table_summary(conn, FORMAL_TABLE)
        target_summary = _table_summary(conn, TARGET_TABLE)

    report = {
        "generated_at": generated_at,
        "actor": "model-agent",
        "candidate_id": "model_agent_10d_no_star_r3gate991_r5gate98_ext_20260624",
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "db_path": str(MODEL_DB),
        "source_table": SOURCE_TABLE,
        "formal_fallback_table": FORMAL_TABLE,
        "target_table": TARGET_TABLE,
        "target_date": TARGET_DATE,
        "method": "history_from_no_star_candidate_latest_day_formal_fallback",
        "method_note": (
            "原 no-star 候选依赖的门控来源表只覆盖到 20260622；"
            "20260623 没有可复用的同口径门控输入，因此最新一日仅用于研究覆盖补齐，"
            "使用当前 formal 10D 分数兜底，不把 20260623 声称为同公式门控结果。"
        ),
        "source_summary": source_summary,
        "formal_summary": formal_summary,
        "target_summary": target_summary,
        "governance": {
            "no_training": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
            "promotion_requires_user_and_audit_confirmation": True,
        },
    }

    (OUT_DIR / "extend_10d_no_star_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_role": "l4_research_prediction_asset",
                "approval_status": "research_only_not_approved_for_l4_or_l5",
                "source_type": "sqlite_table",
                "label": LABEL_COL,
                "candidate_id": report["candidate_id"],
                "db_path": "../MODEL_PREDICTIONS.db",
                "table": TARGET_TABLE,
                "market_db_path": "../../STOCK_DAILY_DATA.db",
                "generated_at": generated_at,
                "row_count": target_summary["row_count"],
                "trade_days": target_summary["trade_days"],
                "stock_count": target_summary["stock_count"],
                "min_trade_date": target_summary["min_trade_date"],
                "max_trade_date": target_summary["max_trade_date"],
                "duplicate_keys": target_summary["duplicate_key_groups"],
                "null_pred_prob": target_summary["null_pred_prob"],
                "notes": "10D no-star 研究候选扩展版：历史段沿用 no-star 候选，20260623 因门控来源缺失使用 formal 10D 兜底，仅作研究覆盖补齐。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# 10D no-star 研究候选扩展报告 20260624",
        "",
        "## 当前结论",
        "",
        f"- 新研究表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 覆盖范围：`{target_summary['min_trade_date']}` 到 `{target_summary['max_trade_date']}`，共 `{target_summary['row_count']}` 行。",
        f"- 最新日 `{TARGET_DATE}` 行数：`{target_summary['target_date_rows']}`。",
        "- 本次未训练模型、未修改生产 manifest、未生成交易信号、未跑回测。",
        "",
        "## 口径说明",
        "",
        "- `20240604-20260622`：沿用 `10d_no_star_r3gate991_r5gate98` 研究候选。",
        "- `20260623`：因原门控来源表缺少当日数据，使用当前 formal 10D 分数兜底，只用于保持研究资产覆盖完整。",
        "- 因此该扩展表不能直接声称为完整同公式门控产物，后续若要发布必须重新审计。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'extend_10d_no_star_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
    ]
    (OUT_DIR / "extend_10d_no_star_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
DB_PATH = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_exante_score_state_asset_v69_20260627"
SCAN_DIR = DATA_DIR / "reports" / "model_agent_3d_exante_score_state_gate_v68_20260627"
BASE_TABLE = "stock_predict_data_model_agent_3d_recent_top_daygate_best_20260627_executable_3d_open_return_research"
CAND_TABLE = "stock_predict_data_model_agent_3d_stability_gate_v57_20260627_executable_3d_open_return_research"
TARGET_TABLE = "stock_predict_data_model_agent_3d_exante_std_gate_v69_20260627_executable_3d_open_return_research"
BASE_STD_THRESHOLD = 0.288675083633
CAND_STD_THRESHOLD = 0.28867511379


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_table(table: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(f"select trade_date, stock_code, pred_prob from {table}", conn)
    df["trade_date"] = df["trade_date"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    return df


def db_summary(table: str) -> dict[str, object]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {table}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from {table}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
        latest = conn.execute(
            f"""
            select trade_date, count(*) rows, count(distinct stock_code) stocks
            from {table}
            group by trade_date
            order by trade_date desc
            limit 5
            """
        ).fetchall()
    return {
        "table": table,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [[str(d), int(r), int(s)] for d, r, s in latest],
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(SCAN_DIR / "exante_score_state_features.csv")
    features["trade_date"] = features["trade_date"].astype(str)
    mask = (features["base_score_std"] >= BASE_STD_THRESHOLD) & (features["cand_score_std"] >= CAND_STD_THRESHOLD)
    active_dates = set(features.loc[mask, "trade_date"])

    base = read_table(BASE_TABLE)
    cand = read_table(CAND_TABLE)
    merged = base.merge(cand, on=["trade_date", "stock_code"], suffixes=("_base", "_cand"), validate="one_to_one")
    use_cand = merged["trade_date"].isin(active_dates)
    out = pd.DataFrame(
        {
            "trade_date": merged["trade_date"],
            "stock_code": merged["stock_code"],
            "pred_prob": merged["pred_prob_base"].where(~use_cand, merged["pred_prob_cand"]),
        }
    )

    with sqlite3.connect(DB_PATH) as conn:
        out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"create index if not exists idx_{TARGET_TABLE}_date_code on {TARGET_TABLE}(trade_date, stock_code)")

    scan_summary = json.loads((SCAN_DIR / "exante_score_state_gate_summary.json").read_text(encoding="utf-8"))
    formula = f"use candidate pred_prob when base_score_std >= {BASE_STD_THRESHOLD} and cand_score_std >= {CAND_STD_THRESHOLD}; otherwise base pred_prob"
    summary = {
        "generated_at": now_iso(),
        "scope": "research_only_3d_exante_std_gate_v69_asset",
        "label": "executable_3d_open_return",
        "base_table": BASE_TABLE,
        "candidate_table": CAND_TABLE,
        "target_table": TARGET_TABLE,
        "formula": formula,
        "feature_source": str(SCAN_DIR / "exante_score_state_features.csv"),
        "active_days": int(len(active_dates)),
        "scan_best": scan_summary.get("best"),
        "db_summary": db_summary(TARGET_TABLE),
        "boundaries": {
            "research_only": True,
            "research_prediction_table_write_only": True,
            "no_training": True,
            "no_formal_manifest_change": True,
            "no_production_manifest_change": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    (REPORT_DIR / "asset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(
            {
                "approval_status": "research_only_not_approved_for_l4_or_l5",
                "asset_role": "research_l4_candidate",
                "source_type": "sqlite_table",
                "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
                "table": TARGET_TABLE,
                "label": "executable_3d_open_return",
                "generated_at": summary["generated_at"],
                "allowed_for_main_workflow": False,
                "notes": "研究候选资产，仅用于模型侧评价和送审准备；未批准进入 formal 或 L5。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = f"""# 3D v69 前视 STD 门控研究资产

生成时间：{summary['generated_at']}

## 结论

已生成 research-only L4 候选表：`{TARGET_TABLE}`。

公式：`{formula}`。

覆盖：`{summary['db_summary']['min_trade_date']}` 至 `{summary['db_summary']['max_trade_date']}`，共 `{summary['db_summary']['trade_days']}` 个交易日，`{summary['db_summary']['row_count']}` 行。

## 边界

- 未训练模型
- 仅写入 research 预测表
- 未修改 formal manifest
- 未修改 production manifest
- 未生成交易信号
- 未运行策略回测
"""
    (REPORT_DIR / "asset_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

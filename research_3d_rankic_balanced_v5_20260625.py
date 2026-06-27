from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "quant" / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
REPORT_DIR = DATA_DIR / "reports" / "model_agent_3d_rankic_balanced_v5_research_20260625"

LABEL = "executable_3d_open_return"
TARGET_TABLE = "stock_predict_data_model_agent_3d_rankic_balanced_v5_20260625_executable_3d_open_return_research"
BASE_TABLE = "stock_predict_data_model_agent_3d_cross_horizon_guard_v3_20260625_executable_3d_open_return_research"
ALT_TABLE = "stock_predict_data_model_agent_1d_broad_stability_guard_20260625_executable_1d_open_return_research"
ALT_ALIAS = "stable1"
FEATURE = "stable1_top50_score_mean"
OP = ">="
THRESHOLD = 0.995471


def now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_scores(conn: sqlite3.Connection, table: str, alias: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        f"select trade_date, stock_code, pred_prob from {quote(table)} order by trade_date, stock_code",
        conn,
    )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame.rename(columns={"pred_prob": f"{alias}_score"})


def build_asset() -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB, timeout=300) as conn:
        conn.execute("PRAGMA busy_timeout=300000")
        base = read_scores(conn, BASE_TABLE, "base3")
        alt = read_scores(conn, ALT_TABLE, ALT_ALIAS)
        frame = base.merge(alt, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
        frame[f"{ALT_ALIAS}_score"] = frame[f"{ALT_ALIAS}_score"].fillna(frame["base3_score"])
        frame["base3_rank"] = frame.groupby("trade_date")["base3_score"].rank(method="average", pct=True)
        frame[f"{ALT_ALIAS}_rank"] = frame.groupby("trade_date")[f"{ALT_ALIAS}_score"].rank(method="average", pct=True)
        feature = (
            frame.groupby("trade_date", sort=True)
            .apply(
                lambda g: float(
                    g.nlargest(min(50, len(g)), f"{ALT_ALIAS}_rank")[f"{ALT_ALIAS}_score"].mean()
                ),
                include_groups=False,
            )
            .reset_index(name=FEATURE)
        )
        frame = frame.merge(feature, on="trade_date", how="left", validate="many_to_one")
        frame["guard_active"] = frame[FEATURE] >= THRESHOLD
        frame["pred_prob"] = frame["base3_rank"]
        frame.loc[frame["guard_active"], "pred_prob"] = frame.loc[frame["guard_active"], f"{ALT_ALIAS}_rank"]
        frame["score_formula"] = f"if {FEATURE} {OP} {THRESHOLD:.12g} then {ALT_ALIAS}_rank else base3_rank"
        out = frame[
            [
                "trade_date",
                "stock_code",
                "pred_prob",
                "base3_score",
                f"{ALT_ALIAS}_score",
                "base3_rank",
                f"{ALT_ALIAS}_rank",
                FEATURE,
                "guard_active",
                "score_formula",
            ]
        ].rename(columns={"base3_score": "base3", f"{ALT_ALIAS}_score": ALT_ALIAS})
        conn.execute(f"drop table if exists {quote(TARGET_TABLE)}")
        out.to_sql(TARGET_TABLE, conn, if_exists="replace", index=False)
        short = hashlib.sha1(TARGET_TABLE.encode("utf-8")).hexdigest()[:12]
        conn.execute(f"create index if not exists idx_{short}_date_code on {quote(TARGET_TABLE)}(trade_date, stock_code)")
        conn.execute(f"create index if not exists idx_{short}_date_pred on {quote(TARGET_TABLE)}(trade_date, pred_prob desc)")
        conn.commit()
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   sum(case when pred_prob is null then 1 else 0 end)
            from {quote(TARGET_TABLE)}
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*)
            from (
              select trade_date, stock_code, count(*) c
              from {quote(TARGET_TABLE)}
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]
        latest = conn.execute(
            f"""
            select trade_date, count(*), count(distinct stock_code)
            from {quote(TARGET_TABLE)}
            group by trade_date
            order by trade_date desc
            limit 10
            """
        ).fetchall()
    return {
        "table": TARGET_TABLE,
        "base_table": BASE_TABLE,
        "alt_table": ALT_TABLE,
        "formula": out["score_formula"].iloc[0],
        "active_days": int(out.groupby("trade_date")["guard_active"].first().sum()),
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "null_pred_prob": int(row[4] or 0),
        "duplicate_key_groups": int(dup),
        "latest_days": [(str(d), int(r), int(s)) for d, r, s in latest],
    }


def write_outputs(stats: dict) -> None:
    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "model_track": "research",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "source_type": "sqlite_table",
        "db_path": "../../../data_file/model_predictions/MODEL_PREDICTIONS.db",
        "table": TARGET_TABLE,
        "label": LABEL,
        "generated_at": now_iso(),
        "decision": "candidate",
        "baseline_for_scan": "research_3d_cross_horizon_guard_v3",
        "research_hypothesis": "rank_ic_balanced_top_metric_enhancement",
        "asset_stats": stats,
        "promotion_requires_user_confirmation": True,
        "audit_required_before_formal": True,
        "boundaries": {
            "research_only": True,
            "no_training": True,
            "no_tuning_training_parameters": True,
            "no_production_manifest_change": True,
            "no_formal_l4_write": True,
            "no_signal": True,
            "no_backtest": True,
        },
    }
    manifest_path = REPORT_DIR / "rankic_balanced_research_manifest.json"
    report_path = REPORT_DIR / "rankic_balanced_research_report.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 3D RankIC 折中研究报告（20260625）",
        "",
        "## 当前结论",
        "",
        "本资产是 research-only 的 3D 折中候选，用更小的 RankIC 代价换取有限 Top 指标增强。它不是生产候选。",
        "",
        "候选规则：",
        "",
        "```text",
        stats["formula"],
        "```",
        "",
        f"- 候选表：`MODEL_PREDICTIONS.db::{TARGET_TABLE}`",
        f"- 日期范围：`{stats['min_trade_date']}` 到 `{stats['max_trade_date']}`",
        f"- 总行数：`{stats['row_count']}`",
        f"- 最新日：`{stats['latest_days'][0][0]}`，行数 `{stats['latest_days'][0][1]}`，股票数 `{stats['latest_days'][0][2]}`",
        f"- 触发交易日数：`{stats['active_days']}`",
        f"- 空分数：`{stats['null_pred_prob']}`",
        f"- 重复键：`{stats['duplicate_key_groups']}`",
        "",
        "## 边界",
        "",
        "- 未训练模型。",
        "- 未改 production/formal manifest。",
        "- 未写 formal L4 表。",
        "- 未生成交易信号。",
        "- 未运行策略回测。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    stats = build_asset()
    write_outputs(stats)
    print(json.dumps({"report_dir": str(REPORT_DIR), "asset_stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

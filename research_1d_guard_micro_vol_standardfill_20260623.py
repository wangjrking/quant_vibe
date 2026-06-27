from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


QUANT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = QUANT_DIR / "data_file"
MODEL_DB = DATA_DIR / "model_predictions" / "MODEL_PREDICTIONS.db"
FACTOR_DIR = DATA_DIR / "production_factor_parts"
OUT_DIR = DATA_DIR / "reports" / "model_agent_1d_guard_micro_vol_gate092_20250101_20260623"

SOURCE_PARQUET = (
    DATA_DIR
    / "reports"
    / "model_agent_1d_guarded_score_fullwindow_20260623"
    / "guarded_score_1d_best_20240604_20260622.parquet"
)
FORMAL_1D_TABLE = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618"
CURRENT_RESEARCH_TABLE = "stock_predict_data_model_agent_1d_dense_guard_smallcap_vr_20260623_executable_1d_open_return_research"
CURRENT_NEUTRALFILL_TABLE = "stock_predict_data_model_agent_1d_dense_guard_smallcap_vr_neutralfill_20260623_executable_1d_open_return_research"
RESEARCH_TABLE = "stock_predict_data_model_agent_1d_guard_micro_vol_gate092_20250101_20260623_executable_1d_open_return_research"

LABEL_COL = "executable_1d_open_return"
DATE_FROM = "20240604"
DATE_TO = "20260622"
GATE_START = "20250101"
GATE_THRESHOLD = 0.92

WINDOWS = {
    "full": ("20240604", "20260611"),
    "recent126": ("20251201", "20260611"),
    "recent63": ("20260304", "20260611"),
}


def evaluate(frame: pd.DataFrame, score_col: str) -> tuple[dict, pd.DataFrame]:
    data = frame[["trade_date", "stock_code", LABEL_COL, score_col]].dropna().copy()
    daily_rows: list[dict] = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "rank_ic": float(ordered[score_col].corr(ordered[LABEL_COL], method="spearman")),
                "top1": float(ordered.head(1)[LABEL_COL].mean()),
                "top3": float(ordered.head(3)[LABEL_COL].mean()),
                "top5": float(ordered.head(5)[LABEL_COL].mean()),
                "top10": float(ordered.head(10)[LABEL_COL].mean()),
                "top20": float(ordered.head(20)[LABEL_COL].mean()),
                "top50": float(ordered.head(50)[LABEL_COL].mean()),
                "top100": float(ordered.head(100)[LABEL_COL].mean()),
            }
        )
    daily = pd.DataFrame(daily_rows)
    summary = {}
    for name, (date_from, date_to) in WINDOWS.items():
        win = daily[(daily["trade_date"] >= date_from) & (daily["trade_date"] <= date_to)].copy()
        summary[name] = {
            "date_min": date_from,
            "date_max": date_to,
            "trade_days": int(win["trade_date"].nunique()),
            "rows": int(len(frame[(frame["trade_date"] >= date_from) & (frame["trade_date"] <= date_to)])),
            "rank_ic": float(win["rank_ic"].mean()),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
            "top50": float(win["top50"].mean()),
            "top100": float(win["top100"].mean()),
        }
    return summary, daily


def front_score(summary: dict) -> float:
    return (
        summary["rank_ic"] * 0.10
        + summary["top1"] * 3.0
        + summary["top3"] * 2.0
        + summary["top5"] * 1.5
        + summary["top10"]
    )


def load_current_research() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as base_pred, [{LABEL_COL}] as {LABEL_COL} from '{CURRENT_RESEARCH_TABLE}'",
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def load_current_neutralfill() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"select trade_date, stock_code, pred_prob as neutralfill_pred from '{CURRENT_NEUTRALFILL_TABLE}'",
            conn,
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def load_formal_features() -> pd.DataFrame:
    with sqlite3.connect(MODEL_DB) as conn:
        frame = pd.read_sql_query(
            f"""
            select trade_date, stock_code, vol, amount, close, pre_close, industry, industry_encode,
                   atr_qfq, close_rate, turnover_rate, turnover_rate_f, circ_mv, total_mv, volume_ratio
            from '{FORMAL_1D_TABLE}'
            where trade_date >= ? and trade_date <= ?
            """,
            conn,
            params=[DATE_FROM, DATE_TO],
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def load_factor_circ_mv() -> pd.DataFrame:
    dataset = ds.dataset(str(FACTOR_DIR), format="parquet")
    table = dataset.to_table(
        columns=["trade_date", "stock_code", "circ_mv"],
        filter=(ds.field("trade_date") >= DATE_FROM) & (ds.field("trade_date") <= DATE_TO),
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def build_report_lines(
    summary_payload: dict,
    comparison_rows: list[dict],
    raw_null_rows: int,
    circ_mv_missing_rows: int,
    current_research_null_rows: int,
    neutralfill_null_rows: int,
) -> list[str]:
    lines = [
        "# 1D Guard Micro Vol 研究资产补齐报告",
        "",
        "## 当前结论",
        "",
        "原始 `guard_micro_vol_a0p05` 候选的头部收益特征很强，但 raw parquet 自带一批 `pred_prob` 空值。",
        "我先尝试用标准因子链路回补 `circ_mv`，结果发现标准链路同一批行里 `circ_mv` 也缺失，因此不能把 raw 候选直接落成 `null_pred_prob=0` 的完整 research 资产。",
        "最终保留方案是一个完整的 gated hybrid：",
        "",
        f"- 若 `raw_guard_pred_prob` 非空且 `base_rank >= {GATE_THRESHOLD}`，使用 raw guard 分数",
        f"- 否则回退到 `neutralfill` 版 1D research：`{CURRENT_NEUTRALFILL_TABLE}`",
        f"- 若最新交易日仍缺口，再回退到 formal 1D 表 `{FORMAL_1D_TABLE}` 中的 `close_rate`",
        "",
        "## 链路核验",
        "",
        f"- raw 候选公式复核最大绝对误差：`{summary_payload['formula_verification_max_abs_error_on_nonnull_rows']:.12f}`",
        f"- raw 候选空值行数：`{raw_null_rows}`",
        f"- 因子回补后仍缺 `circ_mv` 的行数：`{circ_mv_missing_rows}`",
        f"- 旧 current research 自带空值行数：`{current_research_null_rows}`",
        f"- neutralfill research 空值行数：`{neutralfill_null_rows}`",
        f"- 最终 hybrid 落库空值：`{summary_payload['db_summary']['null_pred_prob']}`",
        f"- 最终 hybrid 重复键：`{summary_payload['db_summary']['duplicate_key_groups']}`",
        "",
        "## 相对当前完整 1D research 基线",
        "",
    ]
    for row_item in comparison_rows:
        lines.extend(
            [
                f"### {row_item['window']}",
                f"- RankIC：`{row_item['base_rank_ic']:.8f} -> {row_item['hybrid_rank_ic']:.8f}`",
                f"- Top1：`{row_item['base_top1']:.8f} -> {row_item['hybrid_top1']:.8f}`",
                f"- Top3：`{row_item['base_top3']:.8f} -> {row_item['hybrid_top3']:.8f}`",
                f"- Top5：`{row_item['base_top5']:.8f} -> {row_item['hybrid_top5']:.8f}`",
                f"- Top10：`{row_item['base_top10']:.8f} -> {row_item['hybrid_top10']:.8f}`",
                f"- FrontScore：`{row_item['base_front_score']:.8f} -> {row_item['hybrid_front_score']:.8f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 治理说明",
            "",
            "- 本轮未训练模型。",
            "- 本轮未修改任何 formal manifest。",
            "- 本轮未生成交易信号，未制定交易规则，未跑回测。",
            "- 新资产仅为 `research_only_not_for_l5`。",
            "",
            "## 证据路径",
            "",
            f"- 汇总 JSON：`{(OUT_DIR / 'standardfill_summary.json').as_posix()}`",
            f"- 对比 CSV：`{(OUT_DIR / 'comparison_vs_current_research.csv').as_posix()}`",
            f"- DB 摘要：`{(OUT_DIR / 'db_summary.json').as_posix()}`",
            f"- research manifest：`{(OUT_DIR / 'research_candidate_manifest.json').as_posix()}`",
            f"- research DB 表：`{MODEL_DB.as_posix()}::{RESEARCH_TABLE}`",
        ]
    )
    return lines


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_candidate = pd.read_parquet(SOURCE_PARQUET)
    raw_candidate["trade_date"] = raw_candidate["trade_date"].astype(str)

    current_research = load_current_research()
    current_neutralfill = load_current_neutralfill()
    formal_features = load_formal_features()
    factor_circ_mv = load_factor_circ_mv().rename(columns={"circ_mv": "circ_mv_factor"})

    frame = raw_candidate.merge(current_research, on=["trade_date", "stock_code", LABEL_COL], how="left", validate="one_to_one")
    frame = frame.merge(current_neutralfill, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame = frame.merge(formal_features, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    frame = frame.merge(factor_circ_mv, on=["trade_date", "stock_code"], how="left", validate="one_to_one")

    # 验证原候选公式；这里只用于说明 raw 候选确实来自既有 guard_micro_vol 公式。
    frame["circ_mv_final"] = frame["circ_mv"].fillna(frame["circ_mv_factor"])
    frame["rank_vol"] = frame.groupby("trade_date")["vol"].rank(method="average", pct=True)
    frame["rank_circ_mv"] = frame.groupby("trade_date")["circ_mv_final"].rank(method="average", pct=True)
    frame["risk_micro_vol_rebuilt"] = ((1 - frame["rank_circ_mv"]) + frame["rank_vol"]) / 2
    exact_mask = frame["pred_prob"].notna() & frame["risk_micro_vol"].notna()
    rebuilt_formula_error = (
        frame.loc[exact_mask, "base_rank"] - 0.05 * frame.loc[exact_mask, "risk_micro_vol_rebuilt"] - frame.loc[exact_mask, "pred_prob"]
    ).abs()

    raw_null_rows = int(frame["pred_prob"].isna().sum())
    circ_mv_missing_rows = int(frame["circ_mv_final"].isna().sum())
    current_research_null_rows = int(frame["base_pred"].isna().sum())
    neutralfill_null_rows = int(frame["neutralfill_pred"].isna().sum())

    gate_mask = (
        frame["pred_prob"].notna()
        & (frame["trade_date"] >= GATE_START)
        & (frame["base_rank"] >= GATE_THRESHOLD)
    )
    frame["fallback_pred_prob"] = frame["neutralfill_pred"].fillna(frame["base_pred"])
    frame["fallback_pred_prob"] = frame["fallback_pred_prob"].fillna(frame["close_rate"])
    frame["pred_prob_hybrid"] = frame["fallback_pred_prob"]
    frame.loc[gate_mask, "pred_prob_hybrid"] = frame.loc[gate_mask, "pred_prob"]
    frame["gate_applied"] = gate_mask

    raw_summary, _ = evaluate(frame.rename(columns={"pred_prob": "score"}), "score")
    hybrid_summary, hybrid_daily = evaluate(frame.rename(columns={"pred_prob_hybrid": "score"}), "score")
    base_summary, _ = evaluate(frame.rename(columns={"fallback_pred_prob": "score"}), "score")

    comparison_rows = []
    for window in WINDOWS:
        comparison_rows.append(
            {
                "window": window,
                "base_rank_ic": base_summary[window]["rank_ic"],
                "hybrid_rank_ic": hybrid_summary[window]["rank_ic"],
                "delta_rank_ic": hybrid_summary[window]["rank_ic"] - base_summary[window]["rank_ic"],
                "base_top1": base_summary[window]["top1"],
                "hybrid_top1": hybrid_summary[window]["top1"],
                "delta_top1": hybrid_summary[window]["top1"] - base_summary[window]["top1"],
                "base_top3": base_summary[window]["top3"],
                "hybrid_top3": hybrid_summary[window]["top3"],
                "delta_top3": hybrid_summary[window]["top3"] - base_summary[window]["top3"],
                "base_top5": base_summary[window]["top5"],
                "hybrid_top5": hybrid_summary[window]["top5"],
                "delta_top5": hybrid_summary[window]["top5"] - base_summary[window]["top5"],
                "base_top10": base_summary[window]["top10"],
                "hybrid_top10": hybrid_summary[window]["top10"],
                "delta_top10": hybrid_summary[window]["top10"] - base_summary[window]["top10"],
                "base_front_score": front_score(base_summary[window]),
                "hybrid_front_score": front_score(hybrid_summary[window]),
                "delta_front_score": front_score(hybrid_summary[window]) - front_score(base_summary[window]),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "comparison_vs_current_research.csv", index=False, encoding="utf-8-sig")

    output = frame[
        [
            "trade_date",
            "stock_code",
            "pred_prob_hybrid",
            "fallback_pred_prob",
            "pred_prob",
            "base_rank",
            "risk_micro_vol",
            "vol",
            "circ_mv",
            "circ_mv_factor",
            "gate_applied",
            LABEL_COL,
        ]
    ].rename(
        columns={
            "pred_prob_hybrid": "pred_prob",
            "pred_prob": "raw_guard_pred_prob",
        }
    )

    with sqlite3.connect(MODEL_DB) as conn:
        output.to_sql(RESEARCH_TABLE, conn, if_exists="replace", index=False)
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_code "
            f"on '{RESEARCH_TABLE}'(trade_date, stock_code)"
        )
        conn.execute(
            f"create index if not exists idx_{RESEARCH_TABLE}_date_pred "
            f"on '{RESEARCH_TABLE}'(trade_date, pred_prob desc)"
        )
        row = conn.execute(
            f"""
            select count(*), min(trade_date), max(trade_date), count(distinct trade_date),
                   count(distinct stock_code),
                   sum(case when pred_prob is null then 1 else 0 end)
            from '{RESEARCH_TABLE}'
            """
        ).fetchone()
        dup = conn.execute(
            f"""
            select count(*) from (
              select trade_date, stock_code, count(*) c
              from '{RESEARCH_TABLE}'
              group by trade_date, stock_code
              having c > 1
            )
            """
        ).fetchone()[0]

    db_summary = {
        "db_path": str(MODEL_DB),
        "table": RESEARCH_TABLE,
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "stock_count": int(row[4]),
        "null_pred_prob": int(row[5] or 0),
        "duplicate_key_groups": int(dup),
        "gate_start": GATE_START,
        "gate_threshold": GATE_THRESHOLD,
        "raw_candidate_nonnull_rows": int(frame["pred_prob"].notna().sum()),
        "raw_candidate_null_rows": raw_null_rows,
        "circ_mv_missing_rows_after_factor_backfill": circ_mv_missing_rows,
        "current_research_null_rows": current_research_null_rows,
        "neutralfill_null_rows": neutralfill_null_rows,
        "gate_applied_rows": int(frame["gate_applied"].sum()),
        "source_parquet": str(SOURCE_PARQUET),
        "current_research_table": CURRENT_RESEARCH_TABLE,
        "neutralfill_research_table": CURRENT_NEUTRALFILL_TABLE,
        "formal_1d_table": FORMAL_1D_TABLE,
    }
    (OUT_DIR / "db_summary.json").write_text(json.dumps(db_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_id": "model_agent_1d_guard_micro_vol_gate092_20250101_20260623",
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "raw_guard_formula": "base_rank - 0.05 * risk_micro_vol",
        "hybrid_formula": f"if raw_guard_pred_prob is not null and trade_date >= {GATE_START} and base_rank >= {GATE_THRESHOLD}: raw_guard_pred_prob else neutralfill_1d_research_pred_prob; if that is still null, fallback to formal_1d_close_rate",
        "formula_verification_max_abs_error_on_nonnull_rows": float(rebuilt_formula_error.max()) if len(rebuilt_formula_error) else 0.0,
        "raw_candidate_summary": raw_summary,
        "base_summary": base_summary,
        "hybrid_summary": hybrid_summary,
        "comparison_vs_current_research": comparison_rows,
        "db_summary": db_summary,
        "governance_notes": [
            "research only",
            "no training",
            "no formal manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "standardfill_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hybrid_daily.to_csv(OUT_DIR / "hybrid_daily.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "schema_version": 1,
        "asset_role": "l4_research_prediction_asset",
        "approval_status": "research_only_not_for_l5",
        "promotion_requires_user_confirmation": True,
        "source_type": "sqlite_table",
        "label": LABEL_COL,
        "candidate_id": "model_agent_1d_guard_micro_vol_gate092_20250101_20260623",
        "db_path": "../MODEL_PREDICTIONS.db",
        "table": RESEARCH_TABLE,
        "market_db_path": "../../STOCK_DAILY_DATA.db",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": db_summary["row_count"],
        "trade_days": db_summary["trade_days"],
        "stock_count": db_summary["stock_count"],
        "min_trade_date": db_summary["min_trade_date"],
        "max_trade_date": db_summary["max_trade_date"],
        "duplicate_keys": db_summary["duplicate_key_groups"],
        "null_pred_prob": db_summary["null_pred_prob"],
        "gate_start": GATE_START,
        "gate_threshold": GATE_THRESHOLD,
        "base_score_source": f"MODEL_PREDICTIONS.db::{CURRENT_NEUTRALFILL_TABLE}",
        "secondary_fallback_source": f"MODEL_PREDICTIONS.db::{FORMAL_1D_TABLE}::close_rate",
        "raw_guard_source": str(SOURCE_PARQUET),
        "notes": f"1D guarded micro-vol 原候选因标准链路部分 circ_mv 缺失而无法直接全量落库，因此改为 gate{int(GATE_THRESHOLD * 100):03d} 头部接管 + neutralfill research 回退；对最新交易日缺口再用 formal 1D close_rate 补齐。该资产仅供 research 验证，不进入 formal L4/L5。",
    }
    (OUT_DIR / "research_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = build_report_lines(
        summary_payload=summary_payload,
        comparison_rows=comparison_rows,
        raw_null_rows=raw_null_rows,
        circ_mv_missing_rows=circ_mv_missing_rows,
        current_research_null_rows=current_research_null_rows,
        neutralfill_null_rows=neutralfill_null_rows,
    )
    (OUT_DIR / "standardfill_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

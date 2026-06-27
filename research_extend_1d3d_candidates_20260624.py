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
OUT_DIR = DATA_DIR / "reports" / "model_agent_extend_1d3d_research_20260624"

TARGET_DATE = "20260623"

LABEL_1D = "executable_1d_open_return"
LABEL_3D = "executable_3d_open_return"

OLD_1D_DAYGATE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_20260623_executable_1d_open_return_research"
NEW_1D_DAYGATE = "stock_predict_data_model_agent_1d_gate092_new5d040_daygate_ext_20260624_executable_1d_open_return_research"
SOURCE_1D_GATE = "stock_predict_data_model_agent_1d_guard_micro_vol_gate092_20250101_20260623_executable_1d_open_return_research"
FORMAL_1D = "stock_predict_data_model_agent_toprank_latestfactor_20260618_executable_1d_open_return_score_20240604_20260618"
AUX_5D = "stock_predict_data_model_agent_5d_gate6_balanced_bonus_20260622_executable_5d_open_return_research"

OLD_3D_GUARDED = "stock_predict_data_model_agent_3d_guarded_lowvol_20260623_executable_3d_open_return_research"
NEW_3D_GUARDED = "stock_predict_data_model_agent_3d_guarded_lowvol_ext_20260624_executable_3d_open_return_research"
FORMAL_3D = "stock_predict_data_model_agent_best_full_20260623_executable_3d_open_return_formal"


def _read_sql(conn: sqlite3.Connection, query: str, params: tuple = ()) -> pd.DataFrame:
    frame = pd.read_sql_query(query, conn, params=params)
    if "trade_date" in frame.columns:
        frame["trade_date"] = frame["trade_date"].astype(str)
    if "stock_code" in frame.columns:
        frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def _load_factor(columns: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    dataset = ds.dataset(str(FACTOR_DIR), format="parquet")
    table = dataset.to_table(
        columns=columns,
        filter=(ds.field("trade_date") >= date_from) & (ds.field("trade_date") <= date_to),
    )
    frame = table.to_pandas()
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["stock_code"] = frame["stock_code"].astype(str)
    return frame


def _eval_daily(frame: pd.DataFrame, label_col: str, score_col: str = "pred_prob") -> pd.DataFrame:
    data = frame[["trade_date", "stock_code", score_col, label_col]].dropna().copy()
    rows = []
    for trade_date, group in data.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(score_col, ascending=False, kind="mergesort")
        rows.append(
            {
                "trade_date": trade_date,
                "rows": int(len(ordered)),
                "rank_ic": float(ordered[score_col].corr(ordered[label_col], method="spearman")),
                "top1": float(ordered.head(1)[label_col].mean()),
                "top3": float(ordered.head(3)[label_col].mean()),
                "top5": float(ordered.head(5)[label_col].mean()),
                "top10": float(ordered.head(10)[label_col].mean()),
                "top20": float(ordered.head(20)[label_col].mean()),
                "top50": float(ordered.head(50)[label_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summarize_windows(daily: pd.DataFrame) -> dict:
    windows = {
        "full": None,
        "recent126": 126,
        "recent63": 63,
        "recent20": 20,
    }
    out = {}
    for name, count in windows.items():
        win = daily.tail(count) if count else daily
        out[name] = {
            "date_min": str(win["trade_date"].min()),
            "date_max": str(win["trade_date"].max()),
            "trade_days": int(win["trade_date"].nunique()),
            "rank_ic": float(win["rank_ic"].mean()),
            "top1": float(win["top1"].mean()),
            "top3": float(win["top3"].mean()),
            "top5": float(win["top5"].mean()),
            "top10": float(win["top10"].mean()),
            "top20": float(win["top20"].mean()),
            "top50": float(win["top50"].mean()),
        }
    return out


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


def build_3d(conn: sqlite3.Connection) -> dict:
    old = _read_sql(conn, f"select * from '{OLD_3D_GUARDED}'")
    old = old[old["trade_date"] < TARGET_DATE].copy()

    formal = _read_sql(
        conn,
        f"""
        select trade_date, stock_code, pred_prob as formal_3d_score, [{LABEL_3D}] as {LABEL_3D}
        from '{FORMAL_3D}'
        where trade_date = ?
        """,
        (TARGET_DATE,),
    )
    factors = _load_factor(["trade_date", "stock_code", "low", "vol"], TARGET_DATE, TARGET_DATE)
    target = formal.merge(factors, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    target["formal_rank_score"] = target.groupby("trade_date")["formal_3d_score"].rank(method="average", pct=True)
    target["risk_low_vol"] = (
        (1 - target.groupby("trade_date")["low"].rank(method="average", pct=True))
        + target.groupby("trade_date")["vol"].rank(method="average", pct=True)
    ) / 2
    target["pred_prob"] = target["formal_rank_score"] - 0.15 * target["risk_low_vol"]
    target["score_formula"] = "daily_rank(formal_3d_score) - 0.15 * (((1 - daily_rank(low)) + daily_rank(vol)) / 2)"

    target = target[
        [
            "trade_date",
            "stock_code",
            "pred_prob",
            "formal_3d_score",
            "formal_rank_score",
            "low",
            "vol",
            "risk_low_vol",
            "score_formula",
            LABEL_3D,
        ]
    ]
    output = pd.concat([old[target.columns], target], ignore_index=True)
    output.to_sql(NEW_3D_GUARDED, conn, if_exists="replace", index=False)
    conn.execute(f"create index if not exists idx_{NEW_3D_GUARDED}_date_code on '{NEW_3D_GUARDED}'(trade_date, stock_code)")
    conn.execute(f"create index if not exists idx_{NEW_3D_GUARDED}_date_pred on '{NEW_3D_GUARDED}'(trade_date, pred_prob desc)")
    return _table_summary(conn, NEW_3D_GUARDED)


def build_1d(conn: sqlite3.Connection) -> dict:
    old = _read_sql(conn, f"select * from '{OLD_1D_DAYGATE}'")
    old = old[old["trade_date"] < TARGET_DATE].copy()

    formal = _read_sql(
        conn,
        f"""
        select trade_date, stock_code, pred_prob as formal_pred_prob, close_rate, [{LABEL_1D}] as {LABEL_1D},
               vol, circ_mv
        from '{FORMAL_1D}'
        where trade_date = ?
        """,
        (TARGET_DATE,),
    )
    factor = _load_factor(["trade_date", "stock_code", "circ_mv"], TARGET_DATE, TARGET_DATE)
    factor = factor.rename(columns={"circ_mv": "circ_mv_factor"})
    aux5d = _read_sql(
        conn,
        f"select trade_date, stock_code, pred_prob as pred5d from '{AUX_5D}' where trade_date = ?",
        (TARGET_DATE,),
    )

    target = formal.merge(factor, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    target = target.merge(aux5d, on=["trade_date", "stock_code"], how="left", validate="one_to_one")
    target["base_rank"] = target.groupby("trade_date")["formal_pred_prob"].rank(method="average", pct=True)
    circ = target["circ_mv"].fillna(target["circ_mv_factor"])
    target["risk_micro_vol"] = (
        (1 - circ.groupby(target["trade_date"]).rank(method="average", pct=True))
        + target.groupby("trade_date")["vol"].rank(method="average", pct=True)
    ) / 2
    target["source_pred_prob"] = target["base_rank"] - 0.05 * target["risk_micro_vol"]
    target["base_pred_prob"] = target["close_rate"]
    target["rank5d_pct"] = target.groupby("trade_date")["pred5d"].rank(method="average", pct=True)

    base_top10 = target.sort_values("base_pred_prob", ascending=False, kind="mergesort").head(10)
    source_top10 = target.sort_values("source_pred_prob", ascending=False, kind="mergesort").head(10)
    base_codes = set(base_top10["stock_code"])
    newcomers = source_top10[~source_top10["stock_code"].isin(base_codes)].copy()
    newcomer_mean_5d_pct = float(newcomers["rank5d_pct"].mean()) if len(newcomers) else 1.0
    use_source = newcomer_mean_5d_pct >= 0.4
    target["pred_prob"] = target["source_pred_prob"] if use_source else target["base_pred_prob"]
    target["used_source_gate"] = use_source

    target = target[
        [
            "trade_date",
            "stock_code",
            "pred_prob",
            "base_pred_prob",
            "source_pred_prob",
            "used_source_gate",
            LABEL_1D,
        ]
    ]
    output = pd.concat([old[target.columns], target], ignore_index=True)
    output.to_sql(NEW_1D_DAYGATE, conn, if_exists="replace", index=False)
    conn.execute(f"create index if not exists idx_{NEW_1D_DAYGATE}_date_code on '{NEW_1D_DAYGATE}'(trade_date, stock_code)")
    conn.execute(f"create index if not exists idx_{NEW_1D_DAYGATE}_date_pred on '{NEW_1D_DAYGATE}'(trade_date, pred_prob desc)")
    summary = _table_summary(conn, NEW_1D_DAYGATE)
    summary["target_date_rule"] = {
        "newcomer_count": int(len(newcomers)),
        "newcomer_mean_5d_pct": newcomer_mean_5d_pct,
        "used_source_gate": bool(use_source),
    }
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(MODEL_DB) as conn:
        one_summary = build_1d(conn)
        three_summary = build_3d(conn)

        eval_payload = {}
        for label, table in [(LABEL_1D, NEW_1D_DAYGATE), (LABEL_3D, NEW_3D_GUARDED)]:
            frame = _read_sql(conn, f"select trade_date, stock_code, pred_prob, [{label}] as {label} from '{table}'")
            daily = _eval_daily(frame, label)
            daily.to_csv(OUT_DIR / f"{label}_daily_eval.csv", index=False, encoding="utf-8-sig")
            eval_payload[label] = _summarize_windows(daily)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asset_role": "research_l4_candidate_not_formal",
        "approval_status": "research_only_not_approved_for_l4_or_l5",
        "target_date": TARGET_DATE,
        "db_path": str(MODEL_DB),
        "outputs": {
            "1d": one_summary,
            "3d": three_summary,
        },
        "evaluation": eval_payload,
        "governance_notes": [
            "research only",
            "no training",
            "no production manifest change",
            "no signal",
            "no backtest",
        ],
    }
    (OUT_DIR / "extend_1d3d_research_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 1D/3D 研究候选补齐到 20260623",
        "",
        "## 当前结论",
        "",
        "本次只生成 research-only L4 候选表，不修改 production manifest，不生成交易信号，不运行策略回测。",
        "",
        "## 输出表",
        "",
        f"- 1D：`{NEW_1D_DAYGATE}`",
        f"- 3D：`{NEW_3D_GUARDED}`",
        "",
        "## 覆盖核验",
        "",
        f"- 1D 行数：`{one_summary['row_count']}`，日期：`{one_summary['min_trade_date']}` 到 `{one_summary['max_trade_date']}`，`{TARGET_DATE}` 行数：`{one_summary['target_date_rows']}`，空分数：`{one_summary['null_pred_prob']}`，重复键：`{one_summary['duplicate_key_groups']}`",
        f"- 3D 行数：`{three_summary['row_count']}`，日期：`{three_summary['min_trade_date']}` 到 `{three_summary['max_trade_date']}`，`{TARGET_DATE}` 行数：`{three_summary['target_date_rows']}`，空分数：`{three_summary['null_pred_prob']}`，重复键：`{three_summary['duplicate_key_groups']}`",
        "",
        "## 评价说明",
        "",
        "最新交易日 `20260623` 暂无对应未来标签，因此不进入效果评价。评价截止日由标签非空日期决定。",
        "",
        "## 证据路径",
        "",
        f"- `{(OUT_DIR / 'extend_1d3d_research_summary.json').as_posix()}`",
        f"- `{(OUT_DIR / (LABEL_1D + '_daily_eval.csv')).as_posix()}`",
        f"- `{(OUT_DIR / (LABEL_3D + '_daily_eval.csv')).as_posix()}`",
    ]
    (OUT_DIR / "extend_1d3d_research_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

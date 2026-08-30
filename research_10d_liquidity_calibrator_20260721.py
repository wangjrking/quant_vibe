from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "quant/main/config/model_research_10d_liquidity_calibrator_20260721.json"
MANIFEST_PATH = ROOT / "quant/main/config/prediction_manifests/executable_10d_open_return_l4_formal_20260617.json"
FEATURE_DB = ROOT / "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"
LABEL_DB = ROOT / "quant/data_file/production_assets/duckdb/l3_label_current.duckdb"
OUT = ROOT / "quant/data_file/reports/model_agent_10d_liquidity_calibrator_research_20260721"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def metric_query(weight: float, periods: dict[str, list[str]]) -> str:
    period_values = ",\n".join(
        f"({sql_quote(name)}, {sql_quote(bounds[0])}, {sql_quote(bounds[1])})"
        for name, bounds in periods.items()
    )
    score = f"rank_score + {weight:.12f} * (rank_amount - 0.5)"
    return f"""
        WITH periods(period_name, start_date, end_date) AS (VALUES {period_values}),
        ranked AS (
            SELECT
                trade_date,
                stock_code,
                label_value,
                rank_amount,
                rank_label,
                {score} AS candidate_score,
                percent_rank() OVER (
                    PARTITION BY trade_date ORDER BY {score}
                ) AS candidate_rank,
                row_number() OVER (
                    PARTITION BY trade_date ORDER BY {score} DESC, stock_code
                ) AS score_position
            FROM eval_base
        ),
        daily AS (
            SELECT
                p.period_name,
                r.trade_date,
                corr(r.candidate_rank, r.rank_label) AS rank_ic,
                avg(r.label_value) FILTER (WHERE r.score_position <= 1) AS top1,
                avg(r.label_value) FILTER (WHERE r.score_position <= 3) AS top3,
                avg(r.label_value) FILTER (WHERE r.score_position <= 5) AS top5,
                avg(r.rank_amount) FILTER (WHERE r.score_position <= 1) AS top1_amount_pct,
                avg(r.rank_amount) FILTER (WHERE r.score_position <= 3) AS top3_amount_pct
            FROM ranked r
            JOIN periods p ON r.trade_date BETWEEN p.start_date AND p.end_date
            GROUP BY p.period_name, r.trade_date
        )
        SELECT
            period_name,
            count(*) AS trade_days,
            avg(rank_ic) AS rank_ic,
            avg(top1) AS top1,
            avg(top3) AS top3,
            avg(top5) AS top5,
            avg(top1_amount_pct) AS top1_amount_pct,
            avg(top3_amount_pct) AS top3_amount_pct
        FROM daily
        GROUP BY period_name
        ORDER BY period_name
    """


def period_map(frame: pd.DataFrame) -> dict[str, dict]:
    result = {}
    for row in frame.to_dict("records"):
        name = str(row.pop("period_name"))
        result[name] = {key: (None if pd.isna(value) else float(value)) for key, value in row.items()}
        result[name]["trade_days"] = int(result[name]["trade_days"])
    return result


def table_fingerprint(con: duckdb.DuckDBPyConnection, qualified_table: str, value_columns: list[str]) -> dict:
    values = ", ".join(value_columns)
    row = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            min(trade_date) AS min_trade_date,
            max(trade_date) AS max_trade_date,
            count(DISTINCT trade_date) AS trade_days,
            count(*) FILTER (WHERE stock_code LIKE '%.BJ') AS bj_rows,
            sum(hash(trade_date, stock_code, {values}))::VARCHAR AS table_hash_sum
        FROM {qualified_table}
        """
    ).fetchone()
    return {
        "row_count": int(row[0]),
        "min_trade_date": str(row[1]),
        "max_trade_date": str(row[2]),
        "trade_days": int(row[3]),
        "bj_rows": int(row[4]),
        "table_hash_sum": str(row[5]),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if config["status"] != "frozen_research_only":
        raise RuntimeError("research config is not frozen")
    if manifest["source_type"] != "duckdb_table":
        raise RuntimeError("formal source is not DuckDB")

    pred_db = (MANIFEST_PATH.parent / manifest["db_path"]).resolve()
    pred_table = str(manifest["table"])
    OUT.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute(f"ATTACH '{pred_db.as_posix()}' AS pred (READ_ONLY)")
    con.execute(f"ATTACH '{FEATURE_DB.as_posix()}' AS feat (READ_ONLY)")
    con.execute(f"ATTACH '{LABEL_DB.as_posix()}' AS lab (READ_ONLY)")

    input_quality = con.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(*) FILTER (WHERE f.amount IS NULL) AS null_amount,
            count(*) FILTER (WHERE l.executable_10d_open_return IS NULL) AS null_label,
            count(*) FILTER (WHERE p.stock_code LIKE '%.BJ') AS bj_rows
        FROM pred.{pred_table} p
        JOIN feat.prod_l3_production_factor_parts_20260625 f USING (trade_date, stock_code)
        JOIN lab.prod_l3_prediction_label_parts_current l USING (trade_date, stock_code)
        WHERE p.trade_date BETWEEN {sql_quote(config['observation_start'])}
          AND {sql_quote(config['mature_label_end'])}
        """
    ).fetchone()
    # Mature-date label rows can still be null for names without a valid forward
    # executable window. They are excluded from evaluation, never imputed.
    if int(input_quality[1]) != 0 or int(input_quality[3]) != 0:
        raise RuntimeError(f"fail-closed input quality: {input_quality}")

    con.execute(
        f"""
        CREATE TEMP TABLE eval_base AS
        SELECT
            p.trade_date,
            p.stock_code,
            l.executable_10d_open_return::DOUBLE AS label_value,
            percent_rank() OVER (PARTITION BY p.trade_date ORDER BY p.pred_prob) AS rank_score,
            percent_rank() OVER (PARTITION BY p.trade_date ORDER BY f.amount) AS rank_amount,
            percent_rank() OVER (
                PARTITION BY p.trade_date ORDER BY l.executable_10d_open_return
            ) AS rank_label
        FROM pred.{pred_table} p
        JOIN feat.prod_l3_production_factor_parts_20260625 f USING (trade_date, stock_code)
        JOIN lab.prod_l3_prediction_label_parts_current l USING (trade_date, stock_code)
        WHERE p.trade_date BETWEEN {sql_quote(config['observation_start'])}
          AND {sql_quote(config['mature_label_end'])}
          AND p.stock_code NOT LIKE '%.BJ'
          AND p.pred_prob IS NOT NULL
          AND f.amount IS NOT NULL
          AND l.executable_10d_open_return IS NOT NULL
        """
    )

    selection_periods = config["selection_periods"]
    cost = float(config["round_trip_cost"])
    rows = []
    details = {}
    for weight in config["amount_rank_weights"]:
        metrics = period_map(con.execute(metric_query(float(weight), selection_periods)).fetchdf())
        details[str(weight)] = metrics
        train = metrics["train_all"]
        row = {
            "amount_rank_weight": float(weight),
            **{f"train_{key}": value for key, value in train.items()},
            "train_2022h2_top3_net": metrics["train_2022h2"]["top3"] - cost,
            "train_2023_top3_net": metrics["train_2023"]["top3"] - cost,
        }
        rows.append(row)

    scan = pd.DataFrame(rows)
    baseline = scan.loc[scan["amount_rank_weight"] == 0.0].iloc[0]
    rules = config["selection_rules"]
    scan["train_rank_ic_delta"] = scan["train_rank_ic"] - baseline["train_rank_ic"]
    scan["top1_amount_pct_improvement"] = (
        scan["train_top1_amount_pct"] - baseline["train_top1_amount_pct"]
    )
    scan["selection_eligible"] = (
        (scan["amount_rank_weight"] > 0)
        & (scan["train_2022h2_top3_net"] >= rules["train_2022h2_top3_net_min"])
        & (scan["train_2023_top3_net"] >= rules["train_2023_top3_net_min"])
        & (scan["train_rank_ic_delta"] >= rules["train_rank_ic_delta_min"])
        & (
            scan["top1_amount_pct_improvement"]
            >= rules["top1_amount_percentile_improvement_min"]
        )
    )
    scan["selection_objective"] = (
        0.45 * (scan["train_top1"] - cost)
        + 0.35 * (scan["train_top3"] - cost)
        + 0.20 * (scan["train_top5"] - cost)
        + 0.05 * scan["train_rank_ic"]
    )
    scan = scan.sort_values(
        ["selection_eligible", "selection_objective", "amount_rank_weight"],
        ascending=[False, False, True],
    )
    scan.to_csv(OUT / "train_only_scan.csv", index=False, encoding="utf-8-sig")

    eligible = scan[scan["selection_eligible"]]
    selected_weight = None if eligible.empty else float(eligible.iloc[0]["amount_rank_weight"])
    eval_weights = [0.0] if selected_weight is None else [0.0, selected_weight]
    frozen_eval = {}
    for weight in eval_weights:
        frozen_eval[str(weight)] = period_map(
            con.execute(metric_query(weight, config["frozen_evaluation_periods"])).fetchdf()
        )

    admission_checks = {}
    admission_pass = False
    if selected_weight is not None:
        candidate = frozen_eval[str(selected_weight)]
        base = frozen_eval["0.0"]
        gates = config["post_selection_admission_rules"]
        admission_checks = {
            "validation_2024_top3_net": candidate["validation_2024"]["top3"] - cost,
            "validation_2025_top3_net": candidate["validation_2025"]["top3"] - cost,
            "recent60_top3_net": candidate["recent60"]["top3"] - cost,
            "full_rank_ic_delta": candidate["full"]["rank_ic"] - base["full"]["rank_ic"],
            "full_top1_delta": candidate["full"]["top1"] - base["full"]["top1"],
            "full_top3_delta": candidate["full"]["top3"] - base["full"]["top3"],
        }
        admission_pass = (
            admission_checks["validation_2024_top3_net"] >= gates["validation_2024_top3_net_min"]
            and admission_checks["validation_2025_top3_net"] >= gates["validation_2025_top3_net_min"]
            and admission_checks["recent60_top3_net"] >= gates["recent60_top3_net_min"]
            and admission_checks["full_rank_ic_delta"] >= gates["full_rank_ic_delta_min"]
            and admission_checks["full_top1_delta"] >= gates["full_top1_delta_min"]
            and admission_checks["full_top3_delta"] >= gates["full_top3_delta_min"]
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "actor": "model-agent",
        "status": (
            "research_only_candidate_passed_model_gate"
            if admission_pass
            else "blocked_research_evidence_no_candidate"
        ),
        "candidate_id": "research_10d_amount_rank_calibrator_20260721",
        "selected_amount_rank_weight": selected_weight,
        "selection_used_train_only": True,
        "holdout_used_for_selection": False,
        "admission_pass": admission_pass,
        "admission_checks": admission_checks,
        "frozen_evaluation": frozen_eval,
        "input_quality": {
            "joined_rows": int(input_quality[0]),
            "null_amount": int(input_quality[1]),
            "null_label": int(input_quality[2]),
            "bj_rows": int(input_quality[3]),
        },
        "lineage": {
            "config": str(CONFIG_PATH.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256(CONFIG_PATH),
            "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "script_sha256": sha256(Path(__file__).resolve()),
            "formal_manifest": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "formal_manifest_sha256": sha256(MANIFEST_PATH),
            "prediction_db": str(pred_db.relative_to(ROOT)).replace("\\", "/"),
            "prediction_table": pred_table,
            "feature_db": str(FEATURE_DB.relative_to(ROOT)).replace("\\", "/"),
            "feature_table": "prod_l3_production_factor_parts_20260625",
            "label_db": str(LABEL_DB.relative_to(ROOT)).replace("\\", "/"),
            "label_table": "prod_l3_prediction_label_parts_current",
            "prediction_table_fingerprint": table_fingerprint(
                con, f"pred.{pred_table}", ["pred_prob"]
            ),
        },
        "boundaries": config["boundaries"],
        "approved_for_l5": False,
        "allow_next_layer_continue": False,
    }
    (OUT / "research_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [
        "# 10D 流动性条件校准研究",
        "",
        f"- 状态：`{report['status']}`",
        f"- 训练期选择权重：`{selected_weight}`",
        f"- 冻结后准入：`{admission_pass}`",
        "- 候选选择只使用 2022H2-2023，2024-2026 仅作冻结后评价。",
        "- 未训练模型、未改 formal、未生成信号、未运行回测。",
    ]
    (OUT / "研究报告.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "selected_amount_rank_weight": selected_weight,
        "admission_pass": admission_pass,
        "output": str(OUT.relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read-only development-period market-cap diagnostic for strict 10D OOF."""

from __future__ import annotations

import duckdb


ROOT = "quant/data_file/reports/model_agent_expanding_pit_oof_baselines_20260829_r3"
FEATURE_DB = "quant/data_file/production_assets/duckdb/l3_feature_current.duckdb"


def main() -> int:
    connection = duckdb.connect()
    try:
        connection.execute(f"ATTACH '{FEATURE_DB}' AS l3 (READ_ONLY)")
        result = connection.execute(
            f"""
            WITH score AS (
              SELECT fold_id, trade_date, stock_code, target,
                     ntile(10) OVER (PARTITION BY trade_date ORDER BY pred_prob DESC) AS score_decile
              FROM read_parquet('{ROOT}/10d_oof.parquet')
              WHERE label_mature_within_dev
                AND trade_date BETWEEN '20220101' AND '20241231'
            ), joined AS (
              SELECT score.*, ntile(3) OVER (PARTITION BY score.trade_date ORDER BY feature.total_mv) AS cap_tercile
              FROM score
              JOIN l3.prod_l3_production_factor_parts_20260625 AS feature USING (trade_date, stock_code)
              WHERE feature.total_mv IS NOT NULL
            )
            SELECT fold_id, cap_tercile, count(*) AS rows, avg(target) AS mean_target
            FROM joined
            WHERE score_decile = 1
            GROUP BY fold_id, cap_tercile
            ORDER BY fold_id, cap_tercile
            """
        ).fetchdf()
        print(result.to_json(orient="records"))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

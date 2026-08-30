from pathlib import Path

import duckdb


QUANT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = QUANT_ROOT / "data_file" / "reports" / "strategy_agent_active_l4_prod_repro_grid_20260703"
OUT_DB = REPORT_DIR / "active_l4_wide_cache.duckdb"
SCORE_DB = REPORT_DIR / "score_current.duckdb"
MARKET_DB = QUANT_ROOT / "data_file" / "production_assets" / "duckdb" / "l2_stock_daily_data.duckdb"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(OUT_DB))
    con.execute(f"attach '{SCORE_DB.as_posix()}' as sdb (read_only)")
    con.execute(f"attach '{MARKET_DB.as_posix()}' as mdb (read_only)")
    con.execute("drop table if exists active_l4_wide")
    con.execute(
        """
        create table active_l4_wide as
        with m0 as (
            select
                stock_code,
                trade_date,
                name,
                market,
                open,
                close,
                high,
                low,
                pre_close,
                open_qfq,
                close_qfq,
                low_qfq,
                pre_close_qfq,
                pct_chg,
                amount,
                turnover_rate,
                total_mv,
                atr_qfq,
                ST_TYPE,
                ST_TYPE_name,
                row_number() over(partition by stock_code order by trade_date) as rn
            from mdb.STOCK_DAILY_DATA
            where stock_code not like '%.BJ'
        ),
        m as (
            select
                *,
                (open / nullif(pre_close, 0) - 1) * 100.0 as open_gap_raw_pct,
                case
                    when coalesce(ST_TYPE, '') not in ('', '0') then 1
                    when coalesce(ST_TYPE_name, '') not in ('', '0', '正常') then 1
                    when name like 'ST%' or name like '*ST%' then 1
                    else 0
                end as is_st_risk
            from m0
        ),
        score_ranked as (
            select
                trade_date,
                stock_code,
                pred_prob,
                cume_dist() over(partition by trade_date order by pred_prob) as score_pct_rank,
                row_number() over(partition by trade_date order by pred_prob desc) as score_desc_rank
            from sdb.score
            where stock_code not like '%.BJ'
        )
        select
            s.trade_date as signal_date,
            sig.stock_code,
            sig.name,
            sig.market,
            s.pred_prob,
            s.score_pct_rank,
            s.score_desc_rank,
            sig.pct_chg as signal_pct_chg,
            sig.open_gap_raw_pct as signal_open_gap_raw_pct,
            sig.amount as signal_amount,
            sig.turnover_rate as signal_turnover_rate,
            sig.total_mv as signal_total_mv,
            sig.atr_qfq as signal_atr_qfq,
            buy.trade_date as buy_date,
            buy.open as buy_open,
            buy.pre_close as buy_pre_close,
            buy.open_gap_raw_pct as buy_open_gap_raw_pct,
            buy.amount as buy_amount,
            buy.turnover_rate as buy_turnover_rate,
            buy.total_mv as buy_total_mv,
            buy.atr_qfq as buy_atr_qfq,
            case
                when buy.pre_close is not null and buy.open >= buy.pre_close * 1.095 then 1
                else 0
            end as buy_open_limit_up_10pct_like,
            sell1.open / nullif(buy.open, 0) - 1 as ret_h1,
            sell2.open / nullif(buy.open, 0) - 1 as ret_h2,
            sell3.open / nullif(buy.open, 0) - 1 as ret_h3,
            sell5.open / nullif(buy.open, 0) - 1 as ret_h5
        from score_ranked s
        join m sig
          on sig.stock_code = s.stock_code and sig.trade_date = s.trade_date
        join m buy
          on buy.stock_code = sig.stock_code and buy.rn = sig.rn + 1
        left join m sell1
          on sell1.stock_code = sig.stock_code and sell1.rn = sig.rn + 2
        left join m sell2
          on sell2.stock_code = sig.stock_code and sell2.rn = sig.rn + 3
        left join m sell3
          on sell3.stock_code = sig.stock_code and sell3.rn = sig.rn + 4
        left join m sell5
          on sell5.stock_code = sig.stock_code and sell5.rn = sig.rn + 6
        where sig.is_st_risk = 0
          and buy.is_st_risk = 0
          and sig.name not like '%退%'
          and buy.name not like '%退%'
          and buy.open is not null
          and buy.pre_close is not null
          and buy.pre_close > 0
          and buy.open < buy.pre_close * 1.095
          and s.pred_prob is not null
        """
    )
    con.execute("create index if not exists idx_wide_date on active_l4_wide(signal_date)")
    con.execute("create index if not exists idx_wide_buy_date on active_l4_wide(buy_date)")
    summary = con.execute(
        """
        select
            count(*) as rows,
            min(signal_date) as min_signal_date,
            max(signal_date) as max_signal_date,
            min(buy_date) as min_buy_date,
            max(buy_date) as max_buy_date,
            count(distinct signal_date) as signal_days,
            count(distinct stock_code) as stocks,
            sum(case when ret_h1 is null then 1 else 0 end) as null_ret_h1
        from active_l4_wide
        """
    ).fetchdf()
    summary_path = REPORT_DIR / "active_l4_wide_cache_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(summary_path)
    con.close()


if __name__ == "__main__":
    main()

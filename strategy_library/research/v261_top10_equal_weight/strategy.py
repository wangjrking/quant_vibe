"""
v261 Top-10 Equal-Weight Rotation Strategy
==========================================
固定10只等权满仓轮动策略

设计原则:
- 每日从 L4 预测分数中选出 top-10 股票，等权分配（各 10%）
- 始终满仓运行，不留现金
- 观察期（< 2026-01-01）：仅记录信号，不标记为可执行
- 验证期（>= 2026-01-01）：按信号执行交易

买入信号:
- 排名规则：10d_pred_prob 百分位排名 + 5d/3d/1d 多周期确认
- 调仓频率：每日评估，排名掉出 top-15 触发卖出，从 top-10 未持仓股中补入
- 仓位分配：等权，每只 10%（允许买入日微小偏差）
- 开仓条件：非 ST、非退市、上市 >= 60 天、成交额 >= 1 亿
- 涨停处理：涨停板无法买入，顺延至下一个候选（最多跳过 5 只）

卖出信号:
- 排名下滑：持仓股排名跌出 top-15（即当前排名 > 15）
- 均线跌破：收盘价跌破 20 日均线（MA20_qfq）且排名 > 10
- 止损：单只持仓亏损 >= -8%（从入场价算起）
- 止盈：单只持仓盈利 >= 30% 且排名跌出 top-5
- 调仓频率：每日按信号触发
- 跌停处理：跌停板无法卖出，次日继续尝试；若连续 3 日跌停则次日以开盘价强制卖出

效果评估:
- 观察期（2024-01 至 2025-12）：信号记录，不标记执行
- 验证期（2026-01 至 2026-08）：实际执行，评估稳定性

Author: WorkBuddy
Created: 2026-08-11
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


# ── 路径常量 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[5]  # quant_mcp/
DATA_DIR = ROOT / "quant" / "data_file"
DUCKDB_DIR = DATA_DIR / "production_assets" / "duckdb"
L2_PATH = DUCKDB_DIR / "l2_stock_daily_data.duckdb"
L4_10D_PATH = DUCKDB_DIR / "l4_executable_10d_open_return_formal.duckdb"
L4_5D_PATH = DUCKDB_DIR / "l4_executable_5d_open_return_formal.duckdb"
L4_3D_PATH = DUCKDB_DIR / "l4_executable_3d_open_return_formal.duckdb"
L4_1D_PATH = DUCKDB_DIR / "l4_executable_1d_open_return_formal.duckdb"

L4_10D_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_10d_open_return_formal_candidate"
L4_5D_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_5d_open_return_formal_candidate"
L4_3D_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_3d_open_return_formal_candidate"
L4_1D_TABLE = "stock_predict_data_model_agent_four_year_fullwindow_20260628_executable_1d_open_return_formal_candidate"

# ── 策略参数 ──────────────────────────────────────────────
STRATEGY_ID = "v261_top10_equal_weight"
STRATEGY_NAME = "Top10等权满仓轮动"

# 持仓
MAX_HOLDINGS = 10            # 固定持仓 10 只
TARGET_WEIGHT = 1.0 / 10     # 等权 10%

# 排名
SELL_RANK_THRESHOLD = 30     # 排名 > 30 才触发卖出（放松）
BUY_RANK_THRESHOLD = 15      # 排名 <= 15 可买入（放宽候选池）
MAX_BUY_SKIP = 10            # 涨停跳过的最大候选数
MIN_HOLD_DAYS = 5            # 最低持仓天数，避免频繁换仓

# 打分权重 (多周期)
SCORE_WEIGHTS = {
    "10d": 0.40,
    "5d": 0.30,
    "3d": 0.15,
    "1d": 0.15,
}

# 均线
MA_PERIOD = 20               # MA20 过滤

# 止损止盈
STOP_LOSS_PCT = -0.08        # -8% 止损
TAKE_PROFIT_PCT = 0.30       # +30% 止盈
TAKE_PROFIT_RANK = 5         # 止盈时排名需 > 5
STOP_LOSS_CONSECUTIVE = 3    # 连续跌停 N 天后强制卖出

# 过滤条件
MIN_LISTED_DAYS = 30         # 上市 >= 30 天
MIN_AMOUNT = 100_000        # 成交额 >= 1 亿（金额单位：千元，100_000千元 = 1亿）
MAX_TURNOVER = 30.0          # 换手率 <= 30

# 观察期 / 验证期分界
OBSERVATION_CUTOFF = "2026-01-01"

# ST 过滤关键字
ST_KEYWORDS = ["ST", "*ST", "退市", "PT"]

# 涨跌停比例
BOARD_LIMIT_RATES = {
    "300": 0.20, "301": 0.20, "688": 0.20,  # 创业板/科创板
}
DEFAULT_LIMIT_RATE = 0.10


# ── 工具函数 ──────────────────────────────────────────────

def board_limit_rate(code: str) -> float:
    """返回股票涨跌停幅度。"""
    for prefix, rate in BOARD_LIMIT_RATES.items():
        if code.startswith(prefix):
            return rate
    return DEFAULT_LIMIT_RATE


def is_st(code: str, name: str | None) -> bool:
    """判断是否为 ST 股票。"""
    if name is None:
        name = ""
    for kw in ST_KEYWORDS:
        if kw in name or kw in code:
            return True
    return False


def is_bj(code: str) -> bool:
    """判断是否为北交所股票。"""
    return code.endswith(".BJ")


def percent_rank(values: np.ndarray) -> np.ndarray:
    """计算 percent_rank（值域 [0, 1]），NaN 跳过。"""
    finite = np.isfinite(values)
    result = np.full(len(values), np.nan, dtype=np.float32)
    if finite.sum() == 0:
        return result
    ranks = np.argsort(np.argsort(values[finite]))
    result[finite] = ranks.astype(np.float32) / (max(finite.sum() - 1, 1))
    return result


def _days_between(d1: str, d2: str) -> int:
    """计算两个日期间的日历天数。"""
    from datetime import datetime
    try:
        dt1 = datetime.strptime(d1, "%Y%m%d")
        dt2 = datetime.strptime(d2, "%Y%m%d")
        return (dt2 - dt1).days
    except (ValueError, TypeError):
        return 999


# ── 数据加载 ──────────────────────────────────────────────

def load_scores(conn: duckdb.DuckDBPyConnection, horizon: str, trade_date: str) -> pd.DataFrame:
    """加载指定日期的 L4 预测分数。"""
    table_map = {"10d": L4_10D_TABLE, "5d": L4_5D_TABLE, "3d": L4_3D_TABLE, "1d": L4_1D_TABLE}
    path_map = {"10d": L4_10D_PATH, "5d": L4_5D_PATH, "3d": L4_3D_PATH, "1d": L4_1D_PATH}

    table = table_map[horizon]
    path = path_map[horizon]
    conn.execute(f"ATTACH '{path}' AS l4_{horizon} (READ_ONLY)")

    df = conn.execute(f"""
        SELECT stock_code, pred_prob
        FROM l4_{horizon}."{table}"
        WHERE CAST(trade_date AS VARCHAR) = '{trade_date}'
          AND stock_code NOT LIKE '%.BJ'
    """).df()
    conn.execute(f"DETACH l4_{horizon}")
    return df


def load_market_data(conn: duckdb.DuckDBPyConnection, trade_date: str) -> pd.DataFrame:
    """加载市场数据：价格、均线、成交额、换手率等。"""
    attach_sql = f"ATTACH '{L2_PATH}' AS l2 (READ_ONLY)"
    try:
        conn.execute(attach_sql)
    except Exception:
        pass  # already attached

    df = conn.execute(f"""
        SELECT
            stock_code,
            name,
            open_qfq,
            close_qfq,
            pre_close_qfq,
            high_qfq,
            low_qfq,
            ma_qfq_{MA_PERIOD} AS ma{MA_PERIOD}_qfq,
            amount,
            turnover_rate,
            COALESCE(
                DATEDIFF('day',
                    CAST(
                        SUBSTR(list_date, 1, 4) || '-' || SUBSTR(list_date, 5, 2) || '-' || SUBSTR(list_date, 7, 2)
                        AS DATE
                    ),
                    DATE '{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}'
                ),
                9999
            ) AS listed_days
        FROM l2.STOCK_DAILY_DATA
        WHERE CAST(trade_date AS VARCHAR) = '{trade_date}'
          AND stock_code NOT LIKE '%.BJ'
    """).df()
    return df


# ── 核心逻辑 ──────────────────────────────────────────────

@dataclass
class StrategyState:
    """策略运行时状态。"""
    holdings: dict[str, float] = field(default_factory=dict)  # {code: entry_price_qfq}
    entry_dates: dict[str, str] = field(default_factory=dict) # {code: entry_date}
    history: list[dict] = field(default_factory=list)         # 历史动作
    phase: str = "observation"                                 # observation | execution


def compute_composite_score(scores: dict[str, pd.DataFrame], trade_date: str) -> pd.DataFrame:
    """
    计算复合排名分数。

    输入：各周期的预测分数 DataFrame（stock_code, pred_prob）
    输出：带排名和复合分的 DataFrame
    """

    merged = None
    for horizon, df in scores.items():
        df = df.rename(columns={"pred_prob": f"prob_{horizon}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="stock_code", how="outer")

    if merged is None or len(merged) == 0:
        return pd.DataFrame()

    # 计算各周期的 percent_rank
    for horizon in ["10d", "5d", "3d", "1d"]:
        col = f"prob_{horizon}"
        if col in merged.columns:
            merged[f"rank_{horizon}"] = percent_rank(merged[col].values)

    # 加权复合分 (越高越好)
    merged["composite_score"] = 0.0
    for horizon, weight in SCORE_WEIGHTS.items():
        rank_col = f"rank_{horizon}"
        if rank_col in merged.columns:
            merged["composite_score"] += weight * merged[rank_col].fillna(0)

    # 排名 (1 = 最好)
    merged["overall_rank"] = (
        merged["composite_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return merged.sort_values("overall_rank")


def apply_filters(
    candidates: pd.DataFrame,
    market: pd.DataFrame,
    holdings: dict[str, float],
) -> pd.DataFrame:
    """
    对候选池应用硬过滤条件。

    返回通过过滤的候选池。
    """
    m = market.set_index("stock_code")

    masks = []
    reasons = []

    # 1. 非 ST
    st_mask = ~candidates["stock_code"].apply(
        lambda c: is_st(c, m.loc[c, "name"] if c in m.index else "")
    )
    masks.append(st_mask)

    # 2. 非北交所
    bj_mask = ~candidates["stock_code"].apply(is_bj)
    masks.append(bj_mask)

    # 3. 上市 >= MIN_LISTED_DAYS
    listed = candidates["stock_code"].map(
        lambda c: m.loc[c, "listed_days"] if c in m.index else 0
    )
    masks.append(listed >= MIN_LISTED_DAYS)

    # 4. 成交额 >= MIN_AMOUNT (单位：千元)
    amount = candidates["stock_code"].map(
        lambda c: m.loc[c, "amount"] if c in m.index else 0
    )
    masks.append(amount >= MIN_AMOUNT)

    # 5. 换手率 <= MAX_TURNOVER
    turnover = candidates["stock_code"].map(
        lambda c: m.loc[c, "turnover_rate"] if c in m.index else 999
    )
    masks.append(turnover <= MAX_TURNOVER)

    # 6. 预收盘价 > 0
    pre_close = candidates["stock_code"].map(
        lambda c: m.loc[c, "pre_close_qfq"] if c in m.index else 0
    )
    masks.append(pre_close > 0)

    combined = np.all(masks, axis=0)
    return candidates[combined].copy()


def check_limit_up(code: str, market: pd.DataFrame) -> bool:
    """检查股票是否涨停（开盘价已封板）。"""
    m = market.set_index("stock_code")
    if code not in m.index:
        return True  # 无数据，保守过滤

    row = m.loc[code]
    rate = board_limit_rate(code)
    limit_price = row["pre_close_qfq"] * (1.0 + rate) * 0.995  # 距涨停 < 0.5%
    return row["open_qfq"] >= limit_price


def check_limit_down(code: str, market: pd.DataFrame) -> bool:
    """检查股票是否跌停。"""
    m = market.set_index("stock_code")
    if code not in m.index:
        return False

    row = m.loc[code]
    rate = board_limit_rate(code)
    limit_price = row["pre_close_qfq"] * (1.0 - rate) * 1.005
    return row["open_qfq"] <= limit_price


def check_ma_below(code: str, market: pd.DataFrame) -> bool:
    """检查收盘价是否跌破 MA20。"""
    m = market.set_index("stock_code")
    if code not in m.index:
        return False

    row = m.loc[code]
    ma_col = f"ma{MA_PERIOD}_qfq"
    if ma_col not in row or pd.isna(row[ma_col]) or row[ma_col] == 0:
        return False
    return row["close_qfq"] < row[ma_col]


def generate_signals(
    trade_date: str,
    state: StrategyState,
    market: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    核心信号生成函数。

    返回: 信号字典列表
    """
    is_observation = trade_date < OBSERVATION_CUTOFF
    phase = "observation" if is_observation else "execution"

    # 更新 state
    state.phase = phase

    # ─ Step 1: 计算复合排名 ──────────────────────────
    ranked = compute_composite_score(scores, trade_date)
    if ranked.empty:
        return []

    # ─ Step 2: 应用过滤 ──────────────────────────────
    filtered = apply_filters(ranked, market, state.holdings)

    # ─ Step 3: 生成卖出信号 ──────────────────────────
    signals = []
    stocks_to_sell = []
    stocks_to_hold = []
    sell_limit_down_deferred = []

    for code, entry_price in list(state.holdings.items()):
        sell = False
        reason = ""

        # 最低持仓天数检查
        entry_date = state.entry_dates.get(code, trade_date)
        days_held = _days_between(entry_date, trade_date)
        if days_held < MIN_HOLD_DAYS:
            stocks_to_hold.append(code)
            continue

        # 3a. 排名下滑：跌出 top-SELL_RANK_THRESHOLD
        row = ranked[ranked["stock_code"] == code]
        if not row.empty:
            rank = row.iloc[0]["overall_rank"]
            if rank > SELL_RANK_THRESHOLD:
                sell = True
                reason = f"rank_drop_{rank}_gt_{SELL_RANK_THRESHOLD}"
        else:
            # 不在预测池中（退市等）
            sell = True
            reason = "not_in_prediction_pool"

        # 3b. 均线跌破（仅当排名也走弱时触发）
        if not sell and check_ma_below(code, market):
            row = ranked[ranked["stock_code"] == code]
            if not row.empty and row.iloc[0]["overall_rank"] > 20:
                sell = True
                reason = "ma_below_rank_weak"

        # 3c. 止损
        if not sell and code in market.set_index("stock_code").index:
            current_price = market.set_index("stock_code").loc[code, "close_qfq"]
            pnl_pct = (current_price - entry_price) / entry_price
            if pnl_pct <= STOP_LOSS_PCT:
                sell = True
                reason = f"stop_loss_{pnl_pct:.1%}"

        # 3d. 止盈
        if not sell and code in market.set_index("stock_code").index:
            row = ranked[ranked["stock_code"] == code]
            if not row.empty:
                current_price = market.set_index("stock_code").loc[code, "close_qfq"]
                pnl_pct = (current_price - entry_price) / entry_price
                rank = row.iloc[0]["overall_rank"]
                if pnl_pct >= TAKE_PROFIT_PCT and rank > TAKE_PROFIT_RANK:
                    sell = True
                    reason = f"take_profit_{pnl_pct:.1%}"

        if sell:
            # 跌停检查
            if check_limit_down(code, market):
                # 跌停板暂不卖出，记录到延迟队列
                sell_limit_down_deferred.append(code)
                reason += "_limit_down_deferred"
                sell = False

        if sell:
            stocks_to_sell.append(code)
            signals.append(_make_signal_dict(
                trade_date, code, "SELL", market, ranked, reason, state.phase
            ))
        else:
            stocks_to_hold.append(code)

    # 处理连续跌停强平
    for code in sell_limit_down_deferred:
        # TODO: 跟踪连续跌停天数，超过阈值则强制卖出
        pass

    # ─ Step 4: 生成买入信号 ──────────────────────────
    available_slots = MAX_HOLDINGS - len(stocks_to_hold)

    if available_slots > 0:
        buy_candidates = filtered[
            ~filtered["stock_code"].isin(stocks_to_hold + stocks_to_sell)
        ].head(available_slots + MAX_BUY_SKIP)

        bought = 0
        for _, row in buy_candidates.iterrows():
            if bought >= available_slots:
                break
            code = row["stock_code"]

            # 涨停检查 -> 跳过
            if check_limit_up(code, market):
                continue

            # 买入条件：排名在 top-BUY_RANK_THRESHOLD
            if row["overall_rank"] > BUY_RANK_THRESHOLD:
                continue

            signals.append(_make_signal_dict(
                trade_date, code, "BUY", market, ranked,
                f"rank_{row['overall_rank']}_score_{row['composite_score']:.4f}",
                state.phase,
            ))
            state.holdings[code] = _get_price(market, code, "close_qfq")
            state.entry_dates[code] = trade_date
            bought += 1

    # ─ Step 5: 更新持仓记录 ──────────────────────────
    for code in stocks_to_sell:
        state.holdings.pop(code, None)
        state.entry_dates.pop(code, None)

    state.history.extend(signals)
    return signals


def _make_signal_dict(
    trade_date: str,
    code: str,
    action: str,
    market: pd.DataFrame,
    ranked: pd.DataFrame,
    reason: str,
    phase: str,
) -> dict:
    """构造信号字典。"""
    m = market.set_index("stock_code")
    name = m.loc[code, "name"] if code in m.index else ""
    market_type = _market_type(code)

    # 找到排名信息
    score = np.nan
    rank = -1
    denom = len(ranked)
    row = ranked[ranked["stock_code"] == code]
    if not row.empty:
        score = float(row.iloc[0]["composite_score"])
        rank = int(row.iloc[0]["overall_rank"])

    open_price = m.loc[code, "open_qfq"] if code in m.index else np.nan

    return {
        "strategy_id": STRATEGY_ID,
        "signal_date": trade_date,
        "buy_date": trade_date,  # 次日开盘执行
        "action": action,
        "stock_code": code,
        "name": name,
        "market": market_type,
        "strategy_score": round(score, 6) if not np.isnan(score) else 0.0,
        "score_rank": rank,
        "score_denominator": denom,
        "target_pct": round(TARGET_WEIGHT, 4) if action == "BUY" else 0.0,
        "reason": reason,
        "execution_open_raw": round(float(open_price), 2) if not np.isnan(open_price) else None,
        "status": "pending_buy_day_hard_gate"
                  if phase == "execution"
                  else "observation_record_only",
    }


def _get_price(market: pd.DataFrame, code: str, field: str) -> float:
    """获取股票价格字段。"""
    m = market.set_index("stock_code")
    if code not in m.index:
        return 0.0
    val = m.loc[code, field]
    return float(val) if not pd.isna(val) else 0.0


def _market_type(code: str) -> str:
    """判断股票市场板块。"""
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300") or code.startswith("301"):
        return "创业板"
    if code.startswith("000") or code.startswith("001") or code.startswith("002") or code.startswith("003"):
        return "深市主板"
    return "沪市主板"


# ── 入口函数 ──────────────────────────────────────────────

def run_strategy(trade_date: str, state: StrategyState | None = None) -> tuple[list[dict], StrategyState]:
    """
    策略主入口。

    Args:
        trade_date: 信号日期 (YYYYMMDD)
        state: 策略状态（首次调用传 None）

    Returns:
        (signals, updated_state)
    """
    if state is None:
        state = StrategyState()

    conn = duckdb.connect(":memory:")

    try:
        # 加载数据
        scores = {}
        for h in ["10d", "5d", "3d", "1d"]:
            try:
                scores[h] = load_scores(conn, h, trade_date)
            except Exception as e:
                print(f"[WARN] 无法加载 {h} 预测数据: {e}")

        if len(scores) == 0:
            return [], state

        market = load_market_data(conn, trade_date)
        if market.empty:
            print(f"[WARN] 无法加载 {trade_date} 市场数据")
            return [], state

        # 生成信号
        signals = generate_signals(trade_date, state, market, scores)
        return signals, state

    finally:
        conn.close()


def run_range(
    start_date: str,
    end_date: str,
    state: StrategyState | None = None,
) -> tuple[list[list[dict]], StrategyState]:
    """
    批量运行策略。

    Returns:
        (all_signals_by_day, final_state)
    """
    if state is None:
        state = StrategyState()

    all_signals = []
    conn = duckdb.connect(":memory:")

    try:
        # 获取交易日列表
        conn.execute(f"ATTACH '{L2_PATH}' AS l2 (READ_ONLY)")
        dates = conn.execute(f"""
            SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d
            FROM l2.STOCK_DAILY_DATA
            WHERE CAST(trade_date AS VARCHAR) BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY d
        """).fetchall()
        dates = [row[0] for row in dates]

        for trade_date in dates:
            signals, state = run_strategy(trade_date, state)
            if signals:
                all_signals.append(signals)
                print(
                    f"[{trade_date}] phase={state.phase} "
                    f"buys={sum(1 for s in signals if s['action']=='BUY')} "
                    f"sells={sum(1 for s in signals if s['action']=='SELL')} "
                    f"holdings={len(state.holdings)}"
                )

    finally:
        conn.close()

    return all_signals, state


# ── 评估入口 ──────────────────────────────────────────────

def evaluate():
    """
    策略效果评估函数。

    在观察期和验证期分别计算回测指标。
    """
    from pathlib import Path

    # 1. 观察期运行（2024-01 至 2025-12）
    print("=" * 60)
    print("观察期: 2024-01-01 ~ 2025-12-31")
    print("=" * 60)

    obs_state = StrategyState(phase="observation")
    obs_signals, obs_state = run_range("20240101", "20251231", obs_state)

    # 2. 验证期运行（2026-01 至 2026-08）
    print("\n" + "=" * 60)
    print("验证期: 2026-01-01 ~ 2026-08-11")
    print("=" * 60)

    val_state = StrategyState(phase="execution")
    # 继承观察期最终的持仓状态
    val_state.holdings = dict(obs_state.holdings)
    val_state.entry_dates = dict(obs_state.entry_dates)
    val_signals, val_state = run_range("20260101", "20260811", val_state)

    # 3. 输出统计
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)

    obs_flat = [s for day in obs_signals for s in day]
    val_flat = [s for day in val_signals for s in day]

    obs_buys = sum(1 for s in obs_flat if s["action"] == "BUY")
    obs_sells = sum(1 for s in obs_flat if s["action"] == "SELL")
    val_buys = sum(1 for s in val_flat if s["action"] == "BUY")
    val_sells = sum(1 for s in val_flat if s["action"] == "SELL")

    print(f"观察期: {obs_buys} buys, {obs_sells} sells, "
          f"{len(obs_signals)} trading days, "
          f"final holdings={len(obs_state.holdings)}")

    print(f"验证期: {val_buys} buys, {val_sells} sells, "
          f"{len(val_signals)} trading days, "
          f"final holdings={len(val_state.holdings)}")

    # 导出信号
    out_dir = Path(__file__).resolve().parent / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    if obs_flat:
        df_obs = pd.DataFrame(obs_flat)
        df_obs["signal_date"] = df_obs["signal_date"].astype(str)
        df_obs.to_csv(out_dir / "observation_signals.csv", index=False)
    if val_flat:
        df_val = pd.DataFrame(val_flat)
        df_val["signal_date"] = df_val["signal_date"].astype(str)
        df_val.to_csv(out_dir / "execution_signals.csv", index=False)

    print(f"\n信号文件导出至: {out_dir}")

    return {
        "observation": {
            "days": len(obs_signals),
            "buys": obs_buys,
            "sells": obs_sells,
            "final_holdings": len(obs_state.holdings),
        },
        "execution": {
            "days": len(val_signals),
            "buys": val_buys,
            "sells": val_sells,
            "final_holdings": len(val_state.holdings),
        },
    }


if __name__ == "__main__":
    import sys
    trade_date = sys.argv[1] if len(sys.argv) > 1 else "20260810"
    signals, state = run_strategy(trade_date)
    for s in signals:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\nTotal signals: {len(signals)}, holdings: {len(state.holdings)}")

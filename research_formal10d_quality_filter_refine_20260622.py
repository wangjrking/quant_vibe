from __future__ import annotations

import ast
import csv
import datetime as datetime_module
import json
import math
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from gm_signal_module import build_gm_signal_rows, load_market_rows_by_trade_date, write_gm_signals_csv
from selection_module import SelectionConfig


ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "quant" / "main"
MANIFEST_5D = MAIN / "config" / "prediction_manifests" / "executable_5d_open_return_l4_formal_20260620.json"
MANIFEST_10D = MAIN / "config" / "prediction_manifests" / "executable_10d_open_return_l4_formal_20260617.json"
REPORT_DIR = (
    ROOT
    / "quant"
    / "data_file"
    / "reports"
    / "strategy_agent_model_application_20260621"
    / "formal10d_quality_filter_refine_20260622"
)
STRATEGY_DIR = Path(r"D:\work\dfcf\juejin\strategy\ce327750-634e-11f1-b8d7-10ffe0295517")
JUEJIN_PYTHON = Path(r"C:\Users\wangj\.conda\envs\my_quant\python.exe")
START_DATE = "20240604"
END_DATE = "20260618"


def _load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if manifest.get("approval_status") != "approved_for_l5":
        raise RuntimeError(f"manifest is not approved_for_l5: {path}")
    if manifest.get("source_type") != "sqlite_table":
        raise RuntimeError(f"manifest is not sqlite_table: {path}")
    return manifest


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    return (manifest_path.parent / str(value)).resolve()


M5 = _load_manifest(MANIFEST_5D)
M10 = _load_manifest(MANIFEST_10D)
PRED_DB = _resolve_manifest_path(MANIFEST_10D, M10["db_path"])
MARKET_DB = _resolve_manifest_path(MANIFEST_10D, M10["market_db_path"])
TABLE_5D = str(M5["table"])
TABLE_10D = str(M10["table"])


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def _to_float(value, default=None):
    if value in (None, "", "None"):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "neg").replace("None", "none").replace("/", "_")


def _rank5d_by_day() -> dict[tuple[str, str], float]:
    conn = _connect_readonly(PRED_DB)
    try:
        rows = conn.execute(
            f"""
            SELECT trade_date, stock_code, pred_prob
            FROM "{TABLE_5D}"
            WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
            ORDER BY trade_date, pred_prob DESC
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["trade_date"]), []).append(row)

    ranks: dict[tuple[str, str], float] = {}
    for trade_date, day_rows in grouped.items():
        denom = max(len(day_rows) - 1, 1)
        for idx, row in enumerate(day_rows):
            ranks[(trade_date, str(row["stock_code"]))] = 1.0 - (idx / denom)
    return ranks


def _load_rows() -> list[dict]:
    rank5d = _rank5d_by_day()
    conn = _connect_readonly(PRED_DB)
    try:
        base_rows = conn.execute(
            f"""
            SELECT trade_date, stock_code, pred_prob, close, pre_close, industry, industry_encode,
                   atr_qfq, close_rate, amount, vol, turnover_rate, turnover_rate_f,
                   circ_mv, total_mv, volume_ratio
            FROM "{TABLE_10D}"
            WHERE trade_date >= ? AND trade_date <= ? AND pred_prob IS NOT NULL
            """,
            (START_DATE, END_DATE),
        ).fetchall()
    finally:
        conn.close()
    rows: list[dict] = []
    for row in base_rows:
        item = dict(row)
        key = (str(item["trade_date"]), str(item["stock_code"]))
        item["rank5d"] = rank5d.get(key)
        atr = _to_float(item.get("atr_qfq"))
        close = _to_float(item.get("close"))
        item["atr_ratio"] = (atr / close) if atr is not None and close is not None and close > 0 else None
        rows.append(item)
    return rows


def _passes(row: dict, cfg: dict) -> bool:
    pred = _to_float(row.get("pred_prob"))
    if pred is None:
        return False
    if cfg.get("pred_min") is not None and pred < float(cfg["pred_min"]):
        return False
    if cfg.get("pred_max") is not None and pred > float(cfg["pred_max"]):
        return False
    rank5d = _to_float(row.get("rank5d"))
    if cfg.get("rank5d_min") is not None and (rank5d is None or rank5d < float(cfg["rank5d_min"])):
        return False
    close_rate = _to_float(row.get("close_rate"))
    if cfg.get("close_rate_min") is not None and (close_rate is None or close_rate < float(cfg["close_rate_min"])):
        return False
    if cfg.get("close_rate_max") is not None and (close_rate is None or close_rate > float(cfg["close_rate_max"])):
        return False
    amount = _to_float(row.get("amount"))
    if cfg.get("amount_min") is not None and (amount is None or amount < float(cfg["amount_min"])):
        return False
    if cfg.get("amount_max") is not None and (amount is None or amount > float(cfg["amount_max"])):
        return False
    turnover = _to_float(row.get("turnover_rate"))
    if cfg.get("turnover_min") is not None and (turnover is None or turnover < float(cfg["turnover_min"])):
        return False
    if cfg.get("turnover_max") is not None and (turnover is None or turnover > float(cfg["turnover_max"])):
        return False
    total_mv = _to_float(row.get("total_mv"))
    if cfg.get("total_mv_min") is not None and (total_mv is None or total_mv < float(cfg["total_mv_min"])):
        return False
    if cfg.get("total_mv_max") is not None and (total_mv is None or total_mv > float(cfg["total_mv_max"])):
        return False
    volume_ratio = _to_float(row.get("volume_ratio"))
    if cfg.get("volume_ratio_min") is not None and (volume_ratio is None or volume_ratio < float(cfg["volume_ratio_min"])):
        return False
    if cfg.get("volume_ratio_max") is not None and (volume_ratio is None or volume_ratio > float(cfg["volume_ratio_max"])):
        return False
    atr_ratio = _to_float(row.get("atr_ratio"))
    if cfg.get("atr_ratio_max") is not None and (atr_ratio is None or atr_ratio > float(cfg["atr_ratio_max"])):
        return False
    return True


def _candidate_sort_key(row: dict, sort_mode: str):
    pred = _to_float(row.get("pred_prob"), -999.0)
    amount = _to_float(row.get("amount"), 0.0)
    turnover = _to_float(row.get("turnover_rate"), 0.0)
    total_mv = _to_float(row.get("total_mv"), 999999999.0)
    atr_ratio = _to_float(row.get("atr_ratio"), 999.0)
    close_rate = _to_float(row.get("close_rate"), 1.0)
    rank5d = _to_float(row.get("rank5d"), 0.0)
    if sort_mode == "score_liq_lowvol":
        return (-pred, atr_ratio, -math.log1p(max(amount, 0.0)), total_mv)
    if sort_mode == "score_small_liq":
        return (-pred, total_mv, -math.log1p(max(amount, 0.0)))
    if sort_mode == "score_turn_lowvol":
        return (-pred, -turnover, atr_ratio, total_mv)
    if sort_mode == "balanced":
        quality = pred + 0.08 * rank5d + 0.015 * math.log1p(max(amount, 0.0)) - 0.60 * abs(close_rate - 1.0)
        return (-quality, atr_ratio, total_mv)
    return (-pred, atr_ratio, total_mv)


def _filtered_rows(rows: list[dict], cfg: dict) -> list[dict]:
    selected = [dict(row) for row in rows if _passes(row, cfg)]
    grouped: dict[str, list[dict]] = {}
    for row in selected:
        grouped.setdefault(str(row["trade_date"]), []).append(row)
    ordered: list[dict] = []
    for trade_date, day_rows in grouped.items():
        day_rows.sort(key=lambda row: _candidate_sort_key(row, str(cfg.get("sort_mode") or "score")))
        pool = int(cfg.get("daily_pool") or 80)
        ordered.extend(day_rows[:pool])
    return ordered


def _write_signal(
    rows: list[dict],
    signal_file: Path,
    cfg: dict,
    market_rows: dict[str, dict[str, dict]],
) -> None:
    config = SelectionConfig(
        top_k=int(cfg.get("top_k") or 5),
        pred_col="pred_prob",
        min_pred_prob=None,
        max_atr_ratio=None,
        min_amount=None,
        min_turnover_rate=None,
        max_total_mv=None,
        max_per_industry=999,
        exclude_bj=True,
        exclude_st=True,
        exclude_delisting=True,
        exclude_current_limit=True,
    )
    signals = build_gm_signal_rows(
        rows,
        config=config,
        market_rows_by_trade_date=market_rows,
        holding_days=int(cfg.get("holding_days") or 5),
        max_positions=int(cfg.get("max_positions") or 5),
        weight_mode=str(cfg.get("weight_mode") or "equal"),
        target_total_pct=float(cfg.get("target_total_pct") or 0.98),
        liquidity_target_pct_enabled=bool(cfg.get("liquidity_target_pct_enabled") or False),
        liquidity_min_amount=cfg.get("liquidity_min_amount"),
        liquidity_min_turnover_rate=cfg.get("liquidity_min_turnover_rate"),
        liquidity_mid_scale=float(cfg.get("liquidity_mid_scale") or 0.8),
        liquidity_low_scale=float(cfg.get("liquidity_low_scale") or 0.6),
    )
    by_key = {(str(row["trade_date"]), str(row["stock_code"])): row for row in rows}
    extra_fields = [
        "rank5d",
        "close_rate",
        "amount",
        "turnover_rate",
        "total_mv",
        "circ_mv",
        "volume_ratio",
        "atr_ratio",
    ]
    for signal in signals:
        source = by_key.get((str(signal["signal_date"]), str(signal["stock_code"])), {})
        for field in extra_fields:
            signal[field] = source.get(field)
    write_gm_signals_csv(signals, signal_file)


def _extract_indicator(log_file: Path) -> dict | None:
    if not log_file.exists():
        return None
    marker = "GM_BACKTEST_INDICATOR:"
    for line in reversed(log_file.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            return ast.literal_eval(payload)
        except Exception:
            try:
                return eval(payload, {"__builtins__": {}}, {"datetime": datetime_module})
            except Exception:
                return None
    return None


def _exposure_stats(log_file: Path) -> dict:
    values = []
    active = []
    pattern = re.compile(r"invested_pct=([0-9.]+).*active_positions=(\d+)")
    if not log_file.exists():
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    for match in pattern.finditer(log_file.read_text(encoding="utf-8", errors="ignore")):
        values.append(float(match.group(1)))
        active.append(int(match.group(2)))
    if not values:
        return {"avg_invested_pct": None, "ge80_ratio": None, "max_active_positions": None, "exposure_points": 0}
    return {
        "avg_invested_pct": sum(values) / len(values),
        "ge80_ratio": sum(1 for value in values if value >= 0.80) / len(values),
        "max_active_positions": max(active) if active else None,
        "exposure_points": len(values),
    }


def _signal_stats(signal_file: Path) -> dict:
    with signal_file.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets = [_to_float(row.get("target_pct")) for row in rows]
    targets = [value for value in targets if value is not None]
    return {
        "signal_count": len(rows),
        "buy_days": len({row.get("buy_date") for row in rows if row.get("buy_date")}),
        "avg_signal_target_pct": sum(targets) / len(targets) if targets else None,
        "max_signal_target_pct": max(targets) if targets else None,
    }


def _run_backtest(cfg: dict, signal_file: Path, log_file: Path) -> int:
    if log_file.exists() and _extract_indicator(log_file):
        return 0
    env = os.environ.copy()
    env.update(
        {
            "GM_OPEN_DAILY_SCORE_EXIT": str(int(cfg.get("open_score_exit") or 0)),
            "GM_MAX_DAILY_SELLS": str(int(cfg.get("max_daily_sells") or 1)),
            "GM_LOG_EXPOSURE": "1",
            "GM_SKIP_OPEN_LIMIT_UP_BUY": "1",
            "GM_FORCE_MARKET_ORDER": "0",
            "GM_FORCE_BUY_MARKET_ORDER": "0",
            "GM_SYNC_POSITIONS": str(int(cfg.get("sync") or 0)),
            "GM_CASH_BUFFER": "0.995",
            "GM_VERBOSE_TRADES": "0",
            "GM_SCORE_CONTINUE_ENTRY_RATIO": "none",
        }
    )
    extra_env = cfg.get("extra_env") or {}
    env.update({str(key): str(value) for key, value in extra_env.items()})
    command = [
        str(JUEJIN_PYTHON),
        str(MAIN / "run_juejin_signal_backtest.py"),
        "--strategy-dir",
        str(STRATEGY_DIR),
        "--signal-file",
        str(signal_file),
        "--log-file",
        str(log_file),
        "--max-positions",
        str(int(cfg.get("max_positions") or 5)),
        "--holding-days",
        str(int(cfg.get("holding_days") or 5)),
        "--max-holding-days",
        str(int(cfg.get("max_holding_days") or cfg.get("holding_days") or 5)),
        "--target-position-pct",
        f"{float(cfg.get('target_position_pct') or 0.196):.5f}",
        "--score-db",
        str(PRED_DB),
        "--score-table",
        TABLE_10D,
        "--market-db",
        str(MARKET_DB),
        "--backtest-start",
        "2024-06-05 09:00:00",
        "--backtest-end",
        "2026-06-30 15:30:00",
        "--backtest-adjust",
        "none",
        "--backtest-initial-cash",
        "600000",
        "--backtest-slippage-ratio",
        "0.0015",
    ]
    proc = subprocess.run(command, cwd=str(MAIN), env=env, text=True, capture_output=True)
    if proc.stdout.strip():
        print(proc.stdout.strip().splitlines()[-1], flush=True)
    if proc.returncode != 0 and proc.stderr.strip():
        print(proc.stderr.strip(), flush=True)
    return proc.returncode


def _variant(name: str, **kwargs) -> dict:
    base = {
        "name": name,
        "top_k": 5,
        "daily_pool": 120,
        "holding_days": 5,
        "max_holding_days": 5,
        "max_positions": 5,
        "target_total_pct": 0.98,
        "target_position_pct": 0.196,
        "weight_mode": "equal",
        "pred_min": None,
        "pred_max": None,
        "rank5d_min": 0.50,
        "close_rate_min": 0.965,
        "close_rate_max": 1.085,
        "amount_min": None,
        "amount_max": None,
        "turnover_min": None,
        "turnover_max": None,
        "total_mv_min": None,
        "total_mv_max": 200000.0,
        "volume_ratio_min": None,
        "volume_ratio_max": None,
        "atr_ratio_max": None,
        "sort_mode": "score",
        "open_score_exit": 0,
        "max_daily_sells": 1,
        "sync": 0,
        "extra_env": {},
    }
    base.update(kwargs)
    return base


VARIANTS = [
    _variant("base_formal10d"),
    _variant("amt100m", amount_min=100000.0),
    _variant("amt200m", amount_min=200000.0),
    _variant("amt300m", amount_min=300000.0),
    _variant("turn1_20", turnover_min=1.0, turnover_max=20.0),
    _variant("turn2_18", turnover_min=2.0, turnover_max=18.0),
    _variant("turn3_15", turnover_min=3.0, turnover_max=15.0),
    _variant("mv20_200", total_mv_min=20000.0, total_mv_max=200000.0),
    _variant("mv30_150", total_mv_min=30000.0, total_mv_max=150000.0),
    _variant("mv50_200", total_mv_min=50000.0, total_mv_max=200000.0),
    _variant("volr0p5_8", volume_ratio_min=0.5, volume_ratio_max=8.0),
    _variant("volr0p8_6", volume_ratio_min=0.8, volume_ratio_max=6.0),
    _variant("predcap050", pred_max=0.50),
    _variant("predcap045", pred_max=0.45),
    _variant("predfloor002", pred_min=0.02),
    _variant("rank5d055", rank5d_min=0.55),
    _variant("rank5d060", rank5d_min=0.60),
    _variant("close096_108", close_rate_min=0.96, close_rate_max=1.08),
    _variant("close097_107", close_rate_min=0.97, close_rate_max=1.07),
    _variant("amt100_turn1_mv20_200", amount_min=100000.0, turnover_min=1.0, turnover_max=20.0, total_mv_min=20000.0),
    _variant("amt200_turn2_mv30_150", amount_min=200000.0, turnover_min=2.0, turnover_max=18.0, total_mv_min=30000.0, total_mv_max=150000.0),
    _variant("amt100_volr0p5_8", amount_min=100000.0, volume_ratio_min=0.5, volume_ratio_max=8.0),
    _variant("turn2_volr0p8_6", turnover_min=2.0, turnover_max=18.0, volume_ratio_min=0.8, volume_ratio_max=6.0),
    _variant("balanced_sort", sort_mode="balanced"),
    _variant("liq_lowvol_sort", sort_mode="score_liq_lowvol"),
    _variant("small_liq_sort", sort_mode="score_small_liq"),
    _variant("turn_lowvol_sort", sort_mode="score_turn_lowvol"),
    _variant("rank_weight", weight_mode="rank"),
    _variant("score_weight", weight_mode="score"),
    _variant("top4_full", top_k=4, max_positions=4, target_position_pct=0.245),
    _variant("top3_full", top_k=3, max_positions=3, target_position_pct=0.32667),
    _variant("h6_full", holding_days=6, max_holding_days=6, target_position_pct=0.196),
    _variant("h4_full", holding_days=4, max_holding_days=4, target_position_pct=0.196),
    _variant("sync1", sync=1),
    _variant("eqdd_loose", extra_env={"GM_EQUITY_DD_RISK_MODE": "1", "GM_EQUITY_DD_SOFT_TRIGGER": "0.14", "GM_EQUITY_DD_HARD_TRIGGER": "0.24", "GM_EQUITY_DD_RECOVER_TRIGGER": "0.06", "GM_EQUITY_DD_SOFT_SCALE": "0.92", "GM_EQUITY_DD_HARD_SCALE": "0.75"}),
    _variant("light_stop05_h1", extra_env={"GM_LIGHT_STOP_LOSS_PCT": "0.05", "GM_MIN_HOLDING_DAYS_BEFORE_LIGHT_STOP": "1"}),
    _variant("score_exit095", open_score_exit=1, extra_env={"GM_SCORE_EXIT_ENTRY_RATIO": "0.95", "GM_MIN_HOLDING_DAYS_BEFORE_SCORE_EXIT": "2"}),
]


def _summarize(cfg: dict, signal_file: Path, log_file: Path, returncode: int) -> dict:
    indicator = _extract_indicator(log_file) or {}
    row = {
        **{key: value for key, value in cfg.items() if key != "extra_env"},
        "extra_env_json": json.dumps(cfg.get("extra_env") or {}, ensure_ascii=False, sort_keys=True),
        "returncode": returncode,
        "signal_file": str(signal_file),
        "log_file": str(log_file),
        "annual_return": indicator.get("pnl_ratio_annual"),
        "sharpe": indicator.get("sharp_ratio", indicator.get("sharpe_ratio")),
        "max_drawdown": indicator.get("max_drawdown"),
        "cum_return": indicator.get("pnl_ratio"),
        "win_ratio": indicator.get("win_ratio"),
        "open_count": indicator.get("open_count"),
        "close_count": indicator.get("close_count"),
    }
    row.update(_signal_stats(signal_file))
    row.update(_exposure_stats(log_file))
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric(row: dict, key: str) -> float:
    return _to_float(row.get(key), float("-inf"))


def _write_report(rows: list[dict]) -> None:
    best_annual = max(rows, key=lambda row: _metric(row, "annual_return"))
    best_sharpe = max(rows, key=lambda row: _metric(row, "sharpe"))
    target_hits = [
        row
        for row in rows
        if _metric(row, "annual_return") >= 3.0
        and _metric(row, "sharpe") >= 4.0
        and _metric(row, "avg_invested_pct") >= 0.80
    ]
    report = f"""# formal 10D 质量过滤调参结论

## 结论

本轮使用 `executable_10d_open_return_l4_formal_20260617.json` 和 `executable_5d_open_return_l4_formal_20260620.json` 两个 approved formal L4 manifest。策略侧只读取已发布预测资产，未使用 1D、行业限制、月份过滤、日期过滤或外部数据；候选规则仅围绕市值、成交额、换手率、量比、涨跌幅、5D 排名确认和仓位/卖出参数。

- 候选数量：{len(rows)}
- 达到 `年化 >= 3.0`、`夏普 >= 4.0`、`平均持仓 >= 0.80` 的候选数量：{len(target_hits)}
- 是否建议替换当前 L5：{"是" if target_hits else "否"}

## 最优候选

- 最高年化：`{best_annual["name"]}`，年化 `{_metric(best_annual, "annual_return"):.6f}`，夏普 `{_metric(best_annual, "sharpe"):.6f}`，最大回撤 `{_metric(best_annual, "max_drawdown"):.6f}`，平均持仓 `{_metric(best_annual, "avg_invested_pct"):.6f}`。
- 最高夏普：`{best_sharpe["name"]}`，年化 `{_metric(best_sharpe, "annual_return"):.6f}`，夏普 `{_metric(best_sharpe, "sharpe"):.6f}`，最大回撤 `{_metric(best_sharpe, "max_drawdown"):.6f}`，平均持仓 `{_metric(best_sharpe, "avg_invested_pct"):.6f}`。

## 证据

- 汇总表：`{REPORT_DIR / "summary.csv"}`
- 按夏普排序：`{REPORT_DIR / "summary_by_sharpe.csv"}`
- 候选信号目录：`{REPORT_DIR / "signals"}`
- 掘金日志目录：`{REPORT_DIR / "logs"}`
"""
    (REPORT_DIR / "formal10d质量过滤调参结论.md").write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.joinpath("signals").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    rows = _load_rows()
    market_rows = load_market_rows_by_trade_date(MARKET_DB, START_DATE, END_DATE)
    results: list[dict] = []
    for cfg in VARIANTS:
        name = str(cfg["name"])
        signal_file = REPORT_DIR / "signals" / f"{name}.csv"
        log_file = REPORT_DIR / "logs" / f"{name}.log"
        if not (signal_file.exists() and log_file.exists() and _extract_indicator(log_file)):
            filtered = _filtered_rows(rows, cfg)
            _write_signal(filtered, signal_file, cfg, market_rows)
        returncode = _run_backtest(cfg, signal_file, log_file)
        result = _summarize(cfg, signal_file, log_file, returncode)
        results.append(result)
        _write_csv(REPORT_DIR / "summary.csv", results)
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    _write_csv(REPORT_DIR / "summary_by_sharpe.csv", sorted(results, key=lambda row: _metric(row, "sharpe"), reverse=True))
    _write_csv(REPORT_DIR / "summary_by_annual.csv", sorted(results, key=lambda row: _metric(row, "annual_return"), reverse=True))
    _write_report(results)
    print(json.dumps({"report_dir": str(REPORT_DIR), "variants": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

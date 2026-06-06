from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data_file"
REPORT_DIR = DATA / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from backtest_module import BacktestConfig, read_prediction_rows, run_backtest  # noqa: E402


DB = DATA / "odb.db"
TABLE = "stock_predict_data_10d_yield_rate_oos_1y"
START = "20250604"
END = "20260603"


def configure_fonts():
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    font = FontProperties(fname=str(font_path))
    bold = FontProperties(fname=str(bold_path if bold_path.exists() else font_path))
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["font.sans-serif"] = [font.get_name(), "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return font, bold


def daily_df(result):
    df = pd.DataFrame(result.daily_returns)
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "return", "period_return", "equity", "drawdown"])
    df["date"] = pd.to_datetime(df["trade_date"])
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df


def set_tick_font(ax, font, size=9):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)
        tick.set_fontsize(size)


def style_table(table, font, bold):
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for key, cell in table.get_celld().items():
        cell.get_text().set_fontproperties(font)
        if key[0] == 0:
            cell.set_facecolor("#102033")
            cell.get_text().set_color("white")
            cell.get_text().set_fontproperties(bold)


def make_report():
    font, bold = configure_fonts()
    rows = read_prediction_rows(DB, TABLE, START, END)

    configs = {
        "10日持仓-保守推荐": BacktestConfig(
            top_k=3,
            min_pred_prob=0.01,
            max_atr_ratio=0.05,
            return_col="10d_yield_rate",
            holding_period_days=10,
            start_date=START,
            end_date=END,
        ),
        "10日持仓-全样本最优": BacktestConfig(
            top_k=8,
            min_pred_prob=0.01,
            max_atr_ratio=0.10,
            return_col="10d_yield_rate",
            holding_period_days=10,
            start_date=START,
            end_date=END,
        ),
        "隔日交易-旧默认": BacktestConfig(
            top_k=5,
            min_pred_prob=0.03,
            max_atr_ratio=0.06,
            start_date=START,
            end_date=END,
        ),
        "隔日交易-覆盖型": BacktestConfig(
            top_k=8,
            min_pred_prob=0.01,
            max_atr_ratio=0.10,
            start_date=START,
            end_date=END,
        ),
    }
    results = {name: run_backtest(rows, cfg) for name, cfg in configs.items()}
    daily = {name: daily_df(result) for name, result in results.items()}

    monthly_src = daily["10日持仓-保守推荐"].copy()
    monthly_src["month"] = monthly_src["date"].dt.to_period("M").astype(str)
    monthly_return = monthly_src.groupby("month")["return"].apply(lambda s: (1 + s).prod() - 1).reset_index()

    opt10 = pd.read_csv(DATA / "optimizer_10d_holding10_oos_1y_20250604_20260603.csv")
    optopen = pd.read_csv(DATA / "optimizer_open2_oos_1y_20250604_20260603.csv")
    sel = pd.read_csv(DATA / "selection_default_oos_tuned_20260603.csv")
    train_summary = pd.read_csv(DATA / "rolling_train_summary_10d_oos_1y.csv").iloc[0]

    for frame in (opt10, optopen):
        frame["min_pred_prob"] = frame["min_pred_prob"].fillna("None").astype(str)
        frame["max_atr_ratio"] = frame["max_atr_ratio"].fillna("None").astype(str)
    opt10_top = opt10.sort_values("score", ascending=False).head(8).copy()
    optopen_cov = optopen[optopen["trade_day_count"] >= 100].sort_values("score", ascending=False).head(8).copy()

    rec = results["10日持仓-保守推荐"].metrics
    cover = results["隔日交易-覆盖型"].metrics

    fig = plt.figure(figsize=(18, 24), dpi=180, facecolor="#f5f7fb")
    gs = fig.add_gridspec(8, 6, hspace=0.82, wspace=0.45)

    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#102033"))
    ax.text(0.025, 0.68, "量化选股一年 OOS 回测报告", color="white", fontsize=25, fontproperties=bold, transform=ax.transAxes)
    ax.text(
        0.025,
        0.34,
        f"训练: {train_summary.train_start} 至 {train_summary.train_end}    测试: {train_summary.test_start} 至 {train_summary.test_end}    样本: {int(train_summary.prediction_rows):,} 条预测 / 243 个交易日",
        color="#d8e7ff",
        fontsize=12,
        fontproperties=font,
        transform=ax.transAxes,
    )
    ax.text(
        0.025,
        0.12,
        "核心结论: 当前模型更适合按 10 日持仓使用；隔日交易口径在一年 OOS 中不稳定，覆盖足够时为负收益。",
        color="#ffd166",
        fontsize=13,
        fontproperties=bold,
        transform=ax.transAxes,
    )

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    cards = [
        ("推荐参数", "top_k=3\nmin_pred=0.01\nmax_atr=0.05", "#176b87"),
        ("10日持仓累计收益", f"{rec['cumulative_return'] * 100:.2f}%", "#1b998b"),
        ("10日持仓最大回撤", f"{rec['max_drawdown'] * 100:.2f}%", "#2d6a4f"),
        ("交易覆盖", f"{rec['trade_day_count']:.0f} 天\n{rec['trade_count']:.0f} 笔", "#386641"),
        ("胜率 / IC", f"{rec['win_rate'] * 100:.2f}%\nIC {rec['mean_spearman_ic']:.4f}", "#6a4c93"),
        ("隔日交易覆盖型", f"收益 {cover['cumulative_return'] * 100:.2f}%\n回撤 {cover['max_drawdown'] * 100:.2f}%", "#9d0208"),
    ]
    for i, (title, value, color) in enumerate(cards):
        x = 0.01 + i * 0.165
        ax.add_patch(Rectangle((x, 0.08), 0.155, 0.82, transform=ax.transAxes, color=color, alpha=0.94))
        ax.text(x + 0.012, 0.69, title, color="white", fontsize=11, fontproperties=bold, transform=ax.transAxes)
        ax.text(x + 0.012, 0.26, value, color="white", fontsize=16 if i != 0 else 13, fontproperties=bold, transform=ax.transAxes)

    colors = {
        "10日持仓-保守推荐": "#0077b6",
        "10日持仓-全样本最优": "#00a896",
        "隔日交易-旧默认": "#f77f00",
        "隔日交易-覆盖型": "#d62828",
    }

    ax = fig.add_subplot(gs[2:4, :4])
    for name in colors:
        df = daily[name]
        if not df.empty:
            ax.plot(df["date"], df["equity"], label=name, linewidth=2.2, color=colors[name])
    ax.axhline(1, color="#777", linewidth=0.8)
    ax.set_title("资金曲线对比", fontproperties=bold, fontsize=15)
    ax.set_ylabel("净值", fontproperties=font)
    ax.grid(True, alpha=0.25)
    ax.legend(prop=font, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    set_tick_font(ax, font)

    ax = fig.add_subplot(gs[2:4, 4:])
    for name in ["10日持仓-保守推荐", "隔日交易-覆盖型"]:
        df = daily[name]
        if not df.empty:
            ax.plot(df["date"], df["drawdown"] * 100, label=name, linewidth=2, color=colors[name])
            ax.fill_between(df["date"], df["drawdown"] * 100, 0, alpha=0.12, color=colors[name])
    ax.set_title("回撤曲线", fontproperties=bold, fontsize=15)
    ax.set_ylabel("回撤 %", fontproperties=font)
    ax.grid(True, alpha=0.25)
    ax.legend(prop=font, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)
    set_tick_font(ax, font)

    ax = fig.add_subplot(gs[4, :3])
    bar_colors = ["#2a9d8f" if v >= 0 else "#d62828" for v in monthly_return["return"]]
    ax.bar(monthly_return["month"], monthly_return["return"] * 100, color=bar_colors)
    ax.axhline(0, color="#333", linewidth=0.8)
    ax.set_title("推荐策略月度收益（10日持仓）", fontproperties=bold, fontsize=14)
    ax.set_ylabel("%", fontproperties=font)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    set_tick_font(ax, font, 8)

    monthly_candidates = pd.read_csv(DATA / "backtest_candidate_monthly_oos_1y_20250604_20260603.csv")
    full_rows = monthly_candidates[monthly_candidates["period"] == "full"].copy()
    full_rows["label"] = full_rows["config"].map(
        {
            "label_best_001_010_top8": "10日最优 top8",
            "old_003_006_open2": "隔日旧默认",
            "prev_004_006_open2": "隔日高阈值",
            "robust_001_006_open2_top8": "隔日覆盖0.06",
            "robust_001_010_open2_top8": "隔日覆盖0.10",
        }
    )
    ax = fig.add_subplot(gs[4, 3:])
    ax.barh(full_rows["label"], full_rows["cumulative_return"] * 100, color=["#00a896" if x > 0 else "#d62828" for x in full_rows["cumulative_return"]])
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_title("候选策略一年累计收益", fontproperties=bold, fontsize=14)
    ax.set_xlabel("%", fontproperties=font)
    ax.grid(axis="x", alpha=0.25)
    set_tick_font(ax, font, 8)

    ax = fig.add_subplot(gs[5, :3])
    labels = [f"k{int(r.top_k)} p{r.min_pred_prob} atr{r.max_atr_ratio}" for r in opt10_top.itertuples()]
    y = np.arange(len(labels))
    ax.barh(y, opt10_top["score"], color="#0077b6")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=font, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("10日持仓参数排名（score=收益+回撤）", fontproperties=bold, fontsize=14)
    ax.grid(axis="x", alpha=0.25)
    set_tick_font(ax, font, 8)

    ax = fig.add_subplot(gs[5, 3:])
    labels = [f"k{int(r.top_k)} p{r.min_pred_prob} atr{r.max_atr_ratio}\n{int(r.trade_day_count)}天" for r in optopen_cov.itertuples()]
    y = np.arange(len(labels))
    ax.barh(y, optopen_cov["score"], color="#d62828")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=font, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_title("隔日交易参数排名（至少100交易日）", fontproperties=bold, fontsize=14)
    ax.grid(axis="x", alpha=0.25)
    set_tick_font(ax, font, 8)

    ax = fig.add_subplot(gs[6, :3])
    ax.axis("off")
    ax.set_title("最新默认选股（2026-06-03）", fontproperties=bold, fontsize=14, loc="left")
    table_rows = [
        [int(r["rank"]), r["stock_code"], r["name"], f"{r['pred_prob']:.4f}", f"{r['atr_ratio']:.4f}"]
        for _, r in sel[["rank", "stock_code", "name", "pred_prob", "atr_ratio"]].iterrows()
    ]
    table = ax.table(cellText=table_rows, colLabels=["排名", "代码", "名称", "预测值", "ATR比率"], loc="center", cellLoc="center")
    style_table(table, font, bold)

    ax = fig.add_subplot(gs[6, 3:])
    ax.axis("off")
    notes = [
        "使用方式建议",
        "1. 当前参数服务于 10 日持仓，不建议按隔日卖出执行。",
        "2. 默认每天最多选 3 只；若无满足条件股票，应允许空仓。",
        "3. 回测未完全处理真实成交冲击、停牌、容量、组合重叠持仓。",
        "4. 本次单折 OOS 是一年样本，后续应继续做逐月 walk-forward。",
        "5. 训练日志显示 14000 轮后 RMSE 上升，下一版建议加入 early stopping。",
    ]
    ax.text(0.02, 0.95, notes[0], fontsize=15, fontproperties=bold, va="top")
    for i, line in enumerate(notes[1:], start=1):
        ax.text(0.02, 0.95 - i * 0.15, line, fontsize=11, fontproperties=font, va="top")

    ax = fig.add_subplot(gs[7, :])
    ax.axis("off")
    metric_rows = []
    for name in ["10日持仓-保守推荐", "10日持仓-全样本最优", "隔日交易-旧默认", "隔日交易-覆盖型"]:
        m = results[name].metrics
        metric_rows.append(
            [
                name,
                f"{m['trade_day_count']:.0f}",
                f"{m['trade_count']:.0f}",
                f"{m['cumulative_return'] * 100:.2f}%",
                f"{m['max_drawdown'] * 100:.2f}%",
                f"{m['win_rate'] * 100:.2f}%",
                f"{m['avg_trade_return'] * 100:.2f}%",
            ]
        )
    table = ax.table(cellText=metric_rows, colLabels=["策略", "交易日", "交易笔数", "累计收益", "最大回撤", "胜率", "平均单笔"], loc="center", cellLoc="center")
    style_table(table, font, bold)

    out_png = REPORT_DIR / "backtest_report_oos_1y_20250604_20260603.png"
    out_pdf = REPORT_DIR / "backtest_report_oos_1y_20250604_20260603.pdf"
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    md = REPORT_DIR / "backtest_report_oos_1y_20250604_20260603.md"
    md.write_text(
        f"""# 量化选股一年 OOS 回测报告

测试区间：2025-06-04 至 2026-06-03
训练区间：2020-06-04 至 2025-05-24
预测表：`stock_predict_data_10d_yield_rate_oos_1y`

## 推荐参数

`top_k=3, min_pred=0.01, max_atr=0.05`

## 推荐策略指标（10日持仓）

- 交易日数：{rec['trade_day_count']:.0f}
- 交易笔数：{rec['trade_count']:.0f}
- 累计收益：{rec['cumulative_return'] * 100:.2f}%
- 最大回撤：{rec['max_drawdown'] * 100:.2f}%
- 胜率：{rec['win_rate'] * 100:.2f}%
- 平均单笔收益：{rec['avg_trade_return'] * 100:.2f}%
- mean_spearman_ic：{rec['mean_spearman_ic']:.4f}

## 结论

当前模型更适合按 10 日持仓执行；隔日交易口径在一年 OOS 中不稳定，覆盖足够时为负收益。
""",
        encoding="utf-8",
    )
    return out_png, out_pdf, md


if __name__ == "__main__":
    for path in make_report():
        print(path)

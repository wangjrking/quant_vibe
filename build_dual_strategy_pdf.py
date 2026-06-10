from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data_file" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DOCX = REPORT_DIR / "双策略说明书_20260608.docx"
OUTPUT_PDF = REPORT_DIR / "双策略说明书_20260608.pdf"
CHART_DIR = REPORT_DIR / "strategy_dual_assets"
CHART_DIR.mkdir(parents=True, exist_ok=True)

FONT_REG = Path("C:/Windows/Fonts/msyh.ttc")
FONT_BOLD = Path("C:/Windows/Fonts/msyhbd.ttc")


STRATEGIES = [
    {
        "key": "high_return",
        "title": "收益率最高版",
        "subtitle": "clean 满仓收益版（t099）",
        "annualized": 203.50,
        "cumulative": 410.91,
        "sharpe": 2.0598,
        "drawdown": 14.39,
        "avg_exposure": 85.55,
        "peak_exposure": 99.0,
        "open_count": 353,
        "win_ratio": 49.29,
        "first_buy": "2024-06-06",
        "last_buy": "2026-06-03",
        "log_file": r"D:\work\dfcf\juejin\strategy\fb9d4d71-6198-11f1-8a7e-10ffe0295517\gm_backtest_clean_v2_t099_20260608.log",
        "signal_file": r"D:\work\quant\quant_mcp\quant\data_file\gm_signals_rolling_exec5d_alla_light_top1_none_atr07_hold3_equal_clean_v2_t099.csv",
        "strategy_shape": [
            "滚动训练：quarterly expanding 2y",
            "标签：executable_5d_open_return",
            "股票池：all_a_light",
            "选股：top1 + ATR<=0.07 + hold3",
            "仓位：单笔 0.33，峰值接近 99%",
            "执行：开盘价限价买卖，零 WARN",
        ],
        "purpose": "目标是把资金利用率尽量打满，在保持 clean 执行链的前提下，优先追求更高收益率。",
    },
    {
        "key": "high_sharpe",
        "title": "夏普最高版（加仓后）",
        "subtitle": "双指数 all + MA20 + t099",
        "annualized": 130.52,
        "cumulative": 221.35,
        "sharpe": 2.7345,
        "drawdown": 7.46,
        "avg_exposure": 56.49,
        "peak_exposure": 99.0,
        "open_count": 174,
        "win_ratio": 55.17,
        "first_buy": "2024-09-25",
        "last_buy": "2026-05-26",
        "log_file": r"D:\work\dfcf\juejin\strategy\fb9d4d71-6198-11f1-8a7e-10ffe0295517\gm_backtest_hs300_zz500_all_ma20_top1_atr07_hold3_clean_v1_t099_20260608.log",
        "signal_file": r"D:\work\quant\quant_mcp\quant\data_file\gm_signals_hs300_zz500_all_ma20_top1_atr07_hold3_clean_v1_t099.csv",
        "strategy_shape": [
            "滚动训练：quarterly expanding 2y",
            "标签：executable_5d_open_return",
            "股票池：all_a_light",
            "市场过滤：沪深300 与 中证500 都站上 MA20",
            "选股：top1 + ATR<=0.07 + hold3",
            "仓位：单笔 0.33，峰值接近 99%",
            "执行：开盘价限价买卖，零 WARN",
        ],
        "purpose": "目标是在风险调整后收益最优的前提下，把高夏普主线继续加仓，争取更高收益而不过度伤害 Sharpe。",
    },
]


def set_cell_text(cell, text: str, bold: bool = False, color: RGBColor | None = None, size: int = 10) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_table_borders(table) -> None:
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = OxmlElement(f"w:{edge}")
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), "4")
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), "D9E2F2")
        borders.append(tag)
    tbl_pr.append(borders)


def style_doc(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)


def add_title_block(doc: Document) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("双策略说明书")
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(31, 78, 121)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("收益率最高版 vs 夏普最高版（加仓后）")
    r2.font.name = "Microsoft YaHei"
    r2._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r2.font.size = Pt(12)
    r2.font.color.rgb = RGBColor(89, 89, 89)

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run("口径说明：两版均为 clean 执行链，显式开盘价限价买卖，回测日志 0 WARN，已考虑 0.03% 手续费与 0.10% 滑点。")
    r3.font.name = "Microsoft YaHei"
    r3._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r3.font.size = Pt(9.5)
    r3.font.color.rgb = RGBColor(96, 96, 96)


def load_fonts():
    return (
        ImageFont.truetype(str(FONT_REG), 22),
        ImageFont.truetype(str(FONT_REG), 16),
        ImageFont.truetype(str(FONT_REG), 14),
        ImageFont.truetype(str(FONT_BOLD), 26),
        ImageFont.truetype(str(FONT_BOLD), 18),
        ImageFont.truetype(str(FONT_BOLD), 15),
    )


def draw_bar_chart(title: str, subtitle: str, rows: list[tuple[str, float, float]], value_suffix: str, path: Path, higher_better: bool = True) -> None:
    reg22, reg16, reg14, bold26, bold18, bold15 = load_fonts()
    width, height = 1200, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=22, outline="#D9E2F2", width=3, fill="#FAFCFF")
    draw.text((42, 36), title, font=bold26, fill="#1F4E79")
    draw.text((42, 82), subtitle, font=reg16, fill="#666666")

    max_val = max(max(a, b) for _, a, b in rows)
    chart_left = 310
    chart_right = width - 60
    bar_w_max = chart_right - chart_left
    y = 150
    colors = ["#2E75B6", "#70AD47"]
    for label, a, b in rows:
        draw.text((42, y + 10), label, font=bold15, fill="#333333")
        for idx, val in enumerate((a, b)):
            top = y + idx * 34
            draw.rounded_rectangle((chart_left, top, chart_right, top + 22), radius=8, fill="#E9EEF7")
            fill_w = int((val / max_val) * bar_w_max) if max_val else 0
            draw.rounded_rectangle((chart_left, top, chart_left + fill_w, top + 22), radius=8, fill=colors[idx])
            txt = f"{val:.2f}{value_suffix}"
            tw = draw.textbbox((0, 0), txt, font=reg14)[2]
            tx = min(chart_left + fill_w + 10, chart_right - tw)
            draw.text((tx, top + 2), txt, font=reg14, fill="#222222")
        y += 86

    legend_y = height - 58
    draw.rounded_rectangle((46, legend_y, 70, legend_y + 20), radius=6, fill=colors[0])
    draw.text((80, legend_y - 1), "收益率最高版", font=reg14, fill="#333333")
    draw.rounded_rectangle((250, legend_y, 274, legend_y + 20), radius=6, fill=colors[1])
    draw.text((284, legend_y - 1), "夏普版加仓后", font=reg14, fill="#333333")

    if not higher_better:
        note = "此图为风险指标，越低越好。"
        draw.text((width - 280, legend_y - 1), note, font=reg14, fill="#8A4B08")
    img.save(path)


def build_charts() -> list[Path]:
    annual_png = CHART_DIR / "annual_return_compare.png"
    draw_bar_chart(
        "回测核心指标对比",
        "先看收益能力：收益版追求更高年化，夏普版加仓后在收益与稳定之间取平衡。",
        [
            ("年化收益率", STRATEGIES[0]["annualized"], STRATEGIES[1]["annualized"]),
            ("累计收益率", STRATEGIES[0]["cumulative"], STRATEGIES[1]["cumulative"]),
            ("夏普比率", STRATEGIES[0]["sharpe"], STRATEGIES[1]["sharpe"]),
        ],
        "%",
        annual_png,
        higher_better=True,
    )

    risk_png = CHART_DIR / "risk_exposure_compare.png"
    draw_bar_chart(
        "风险与仓位利用率",
        "再看风险侧：收益版更接近持续高暴露，夏普版加仓后依然受市场过滤约束。",
        [
            ("最大回撤", STRATEGIES[0]["drawdown"], STRATEGIES[1]["drawdown"]),
            ("平均持仓率", STRATEGIES[0]["avg_exposure"], STRATEGIES[1]["avg_exposure"]),
            ("峰值持仓率", STRATEGIES[0]["peak_exposure"], STRATEGIES[1]["peak_exposure"]),
        ],
        "%",
        risk_png,
        higher_better=False,
    )

    trade_png = CHART_DIR / "trade_count_compare.png"
    draw_bar_chart(
        "交易节奏与命中率",
        "交易频次反映策略活跃程度，胜率则帮助判断信号的稳定性与市场过滤效果。",
        [
            ("开仓次数", STRATEGIES[0]["open_count"], STRATEGIES[1]["open_count"]),
            ("胜率", STRATEGIES[0]["win_ratio"], STRATEGIES[1]["win_ratio"]),
        ],
        "%",
        trade_png,
        higher_better=True,
    )
    return [annual_png, risk_png, trade_png]


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(16 if level == 1 else 13)
    run.font.color.rgb = RGBColor(31, 78, 121 if level == 1 else 96)
    p.space_before = Pt(8)
    p.space_after = Pt(4)


def add_bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(item)
        r.font.name = "Microsoft YaHei"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        r.font.size = Pt(10.5)


def add_strategy_card(doc: Document, strategy: dict) -> None:
    add_heading(doc, f"{strategy['title']}：{strategy['subtitle']}", level=1)
    p = doc.add_paragraph()
    r = p.add_run(strategy["purpose"])
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.size = Pt(10.5)

    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.autofit = True
    set_table_borders(table)
    hdr = table.rows[0].cells
    set_cell_text(hdr[0], "字段", bold=True, color=RGBColor(255, 255, 255), size=10)
    set_cell_text(hdr[1], "内容", bold=True, color=RGBColor(255, 255, 255), size=10)
    for c in hdr:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E79")
        c._tc.get_or_add_tcPr().append(shading)
    rows = [
        ("训练方式", "rolling / quarterly expanding / 2y"),
        ("预测标签", "executable_5d_open_return"),
        ("股票池", "all_a_light"),
        ("执行方式", "次日开盘价限价买入，到期日开盘价限价卖出，开盘涨停跳过买入，开盘跌停顺延卖出"),
        ("回测口径", "手续费 0.03% + 滑点 0.10%，官方 clean 回测日志 0 WARN"),
        ("信号周期", f"{strategy['first_buy']} 至 {strategy['last_buy']}"),
        ("日志文件", strategy["log_file"]),
        ("信号文件", strategy["signal_file"]),
    ]
    for left, right in rows:
        cells = table.add_row().cells
        set_cell_text(cells[0], left, bold=True)
        set_cell_text(cells[1], right)

    add_heading(doc, "规则拆解", level=2)
    add_bullets(doc, strategy["strategy_shape"])


def build_docx() -> Path:
    charts = build_charts()
    doc = Document()
    style_doc(doc)
    add_title_block(doc)

    add_heading(doc, "核心结论", level=1)
    add_bullets(
        doc,
        [
            "收益率最高版更偏进攻，核心在于不做市场过滤，并把单笔仓位抬到 0.33，从而把总暴露推到接近满仓滚动。",
            "夏普版加仓后更偏稳健，核心在于保留“双指数都站上 MA20 才开仓”的市场环境闸门，同时把单笔仓位也抬到 0.33。",
            "两版都采用相同的 clean 执行链：显式开盘价限价单、开盘涨停跳过、开盘跌停顺延、官方回测日志 0 WARN。",
        ],
    )

    add_heading(doc, "一页对比", level=1)
    comparison = doc.add_table(rows=1, cols=3)
    comparison.style = "Table Grid"
    set_table_borders(comparison)
    headers = comparison.rows[0].cells
    for idx, text in enumerate(["指标", "收益率最高版", "夏普版加仓后"]):
        set_cell_text(headers[idx], text, bold=True, color=RGBColor(255, 255, 255), size=10)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E79")
        headers[idx]._tc.get_or_add_tcPr().append(shading)
    metrics_rows = [
        ("年化收益率", "203.50%", "130.52%"),
        ("累计收益率", "410.91%", "221.35%"),
        ("夏普比率", "2.0598", "2.7345"),
        ("最大回撤", "14.39%", "7.46%"),
        ("平均持仓率", "85.55%", "56.49%"),
        ("峰值持仓率", "99%", "99%"),
        ("开仓次数", "353", "174"),
        ("胜率", "49.29%", "55.17%"),
    ]
    for metric, a, b in metrics_rows:
        row = comparison.add_row().cells
        set_cell_text(row[0], metric, bold=True)
        set_cell_text(row[1], a)
        set_cell_text(row[2], b)

    for chart in charts:
        doc.add_paragraph()
        doc.add_picture(str(chart), width=Inches(6.6))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_strategy_card(doc, STRATEGIES[0])
    add_strategy_card(doc, STRATEGIES[1])

    add_heading(doc, "怎么选", level=1)
    add_bullets(
        doc,
        [
            "如果目标是尽量把收益率做高，而且接受更大的资金暴露与更高的回撤波动，就选收益率最高版。",
            "如果目标是让收益曲线更稳、回撤更低、风险调整后收益更漂亮，就选夏普版加仓后。",
            "收益版更像“持续进攻”，夏普版更像“带市场环境闸门的稳健进攻”。",
        ],
    )

    add_heading(doc, "来源材料", level=1)
    add_bullets(
        doc,
        [
            r"参考 PDF：D:\download\edge\满仓收益版.pdf",
            r"参考 PDF：D:\download\edge\夏普.pdf",
            "说明：这份说明书重新统一了指标口径、执行规则和图表表现，因此最终数字以本说明书列出的官方 clean 日志为准。",
        ],
    )

    doc.save(OUTPUT_DOCX)
    return OUTPUT_DOCX


if __name__ == "__main__":
    path = build_docx()
    print(path)

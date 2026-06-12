from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "data_file" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DOCX = REPORT_DIR / "鍙岀瓥鐣ヨ鏄庝功_20260608.docx"
OUTPUT_PDF = REPORT_DIR / "鍙岀瓥鐣ヨ鏄庝功_20260608.pdf"
CHART_DIR = REPORT_DIR / "strategy_dual_assets"
CHART_DIR.mkdir(parents=True, exist_ok=True)

def find_font(families: list[str]) -> Path | None:
    for family in families:
        try:
            return Path(font_manager.findfont(family, fallback_to_default=False))
        except ValueError:
            continue
    return None


FONT_REG = find_font(["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei"])
FONT_BOLD = find_font(["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei"])


STRATEGIES = [
    {
        "key": "high_return",
        "title": "鏀剁泭鐜囨渶楂樼増",
        "subtitle": "clean 婊′粨鏀剁泭鐗堬紙t099锛?,
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
        "log_file": str(ROOT / "juejin_strategies" / "long_term" / "gm_backtest_clean_v2_t099_20260608.log"),
        "signal_file": str(ROOT / "data_file" / "gm_signals_rolling_exec5d_alla_light_top1_none_atr07_hold3_equal_clean_v2_t099.csv"),
        "strategy_shape": [
            "婊氬姩璁粌锛歲uarterly expanding 2y",
            "鏍囩锛歟xecutable_5d_open_return",
            "鑲＄エ姹狅細all_a_light",
            "閫夎偂锛歵op1 + ATR<=0.07 + hold3",
            "浠撲綅锛氬崟绗?0.33锛屽嘲鍊兼帴杩?99%",
            "鎵ц锛氬紑鐩樹环闄愪环涔板崠锛岄浂 WARN",
        ],
        "purpose": "鐩爣鏄妸璧勯噾鍒╃敤鐜囧敖閲忔墦婊★紝鍦ㄤ繚鎸?clean 鎵ц閾剧殑鍓嶆彁涓嬶紝浼樺厛杩芥眰鏇撮珮鏀剁泭鐜囥€?,
    },
    {
        "key": "high_sharpe",
        "title": "澶忔櫘鏈€楂樼増锛堝姞浠撳悗锛?,
        "subtitle": "鍙屾寚鏁?all + MA20 + t099",
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
        "log_file": str(ROOT / "juejin_strategies" / "long_term" / "gm_backtest_hs300_zz500_all_ma20_top1_atr07_hold3_clean_v1_t099_20260608.log"),
        "signal_file": str(ROOT / "data_file" / "gm_signals_hs300_zz500_all_ma20_top1_atr07_hold3_clean_v1_t099.csv"),
        "strategy_shape": [
            "婊氬姩璁粌锛歲uarterly expanding 2y",
            "鏍囩锛歟xecutable_5d_open_return",
            "鑲＄エ姹狅細all_a_light",
            "甯傚満杩囨护锛氭勃娣?00 涓?涓瘉500 閮界珯涓?MA20",
            "閫夎偂锛歵op1 + ATR<=0.07 + hold3",
            "浠撲綅锛氬崟绗?0.33锛屽嘲鍊兼帴杩?99%",
            "鎵ц锛氬紑鐩樹环闄愪环涔板崠锛岄浂 WARN",
        ],
        "purpose": "鐩爣鏄湪椋庨櫓璋冩暣鍚庢敹鐩婃渶浼樼殑鍓嶆彁涓嬶紝鎶婇珮澶忔櫘涓荤嚎缁х画鍔犱粨锛屼簤鍙栨洿楂樻敹鐩婅€屼笉杩囧害浼ゅ Sharpe銆?,
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
    run = p.add_run("鍙岀瓥鐣ヨ鏄庝功")
    run.bold = True
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(31, 78, 121)

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("鏀剁泭鐜囨渶楂樼増 vs 澶忔櫘鏈€楂樼増锛堝姞浠撳悗锛?)
    r2.font.name = "Microsoft YaHei"
    r2._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r2.font.size = Pt(12)
    r2.font.color.rgb = RGBColor(89, 89, 89)

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run("鍙ｅ緞璇存槑锛氫袱鐗堝潎涓?clean 鎵ц閾撅紝鏄惧紡寮€鐩樹环闄愪环涔板崠锛屽洖娴嬫棩蹇?0 WARN锛屽凡鑰冭檻 0.03% 鎵嬬画璐逛笌 0.10% 婊戠偣銆?)
    r3.font.name = "Microsoft YaHei"
    r3._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r3.font.size = Pt(9.5)
    r3.font.color.rgb = RGBColor(96, 96, 96)


def load_fonts():
    if FONT_REG is None or FONT_BOLD is None:
        default_font = ImageFont.load_default()
        return (default_font, default_font, default_font, default_font, default_font, default_font)
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
    draw.text((80, legend_y - 1), "鏀剁泭鐜囨渶楂樼増", font=reg14, fill="#333333")
    draw.rounded_rectangle((250, legend_y, 274, legend_y + 20), radius=6, fill=colors[1])
    draw.text((284, legend_y - 1), "澶忔櫘鐗堝姞浠撳悗", font=reg14, fill="#333333")

    if not higher_better:
        note = "姝ゅ浘涓洪闄╂寚鏍囷紝瓒婁綆瓒婂ソ銆?
        draw.text((width - 280, legend_y - 1), note, font=reg14, fill="#8A4B08")
    img.save(path)


def build_charts() -> list[Path]:
    annual_png = CHART_DIR / "annual_return_compare.png"
    draw_bar_chart(
        "鍥炴祴鏍稿績鎸囨爣瀵规瘮",
        "鍏堢湅鏀剁泭鑳藉姏锛氭敹鐩婄増杩芥眰鏇撮珮骞村寲锛屽鏅増鍔犱粨鍚庡湪鏀剁泭涓庣ǔ瀹氫箣闂村彇骞宠　銆?,
        [
            ("骞村寲鏀剁泭鐜?, STRATEGIES[0]["annualized"], STRATEGIES[1]["annualized"]),
            ("绱鏀剁泭鐜?, STRATEGIES[0]["cumulative"], STRATEGIES[1]["cumulative"]),
            ("澶忔櫘姣旂巼", STRATEGIES[0]["sharpe"], STRATEGIES[1]["sharpe"]),
        ],
        "%",
        annual_png,
        higher_better=True,
    )

    risk_png = CHART_DIR / "risk_exposure_compare.png"
    draw_bar_chart(
        "椋庨櫓涓庝粨浣嶅埄鐢ㄧ巼",
        "鍐嶇湅椋庨櫓渚э細鏀剁泭鐗堟洿鎺ヨ繎鎸佺画楂樻毚闇诧紝澶忔櫘鐗堝姞浠撳悗渚濈劧鍙楀競鍦鸿繃婊ょ害鏉熴€?,
        [
            ("鏈€澶у洖鎾?, STRATEGIES[0]["drawdown"], STRATEGIES[1]["drawdown"]),
            ("骞冲潎鎸佷粨鐜?, STRATEGIES[0]["avg_exposure"], STRATEGIES[1]["avg_exposure"]),
            ("宄板€兼寔浠撶巼", STRATEGIES[0]["peak_exposure"], STRATEGIES[1]["peak_exposure"]),
        ],
        "%",
        risk_png,
        higher_better=False,
    )

    trade_png = CHART_DIR / "trade_count_compare.png"
    draw_bar_chart(
        "浜ゆ槗鑺傚涓庡懡涓巼",
        "浜ゆ槗棰戞鍙嶆槧绛栫暐娲昏穬绋嬪害锛岃儨鐜囧垯甯姪鍒ゆ柇淇″彿鐨勭ǔ瀹氭€т笌甯傚満杩囨护鏁堟灉銆?,
        [
            ("寮€浠撴鏁?, STRATEGIES[0]["open_count"], STRATEGIES[1]["open_count"]),
            ("鑳滅巼", STRATEGIES[0]["win_ratio"], STRATEGIES[1]["win_ratio"]),
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
    add_heading(doc, f"{strategy['title']}锛歿strategy['subtitle']}", level=1)
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
    set_cell_text(hdr[0], "瀛楁", bold=True, color=RGBColor(255, 255, 255), size=10)
    set_cell_text(hdr[1], "鍐呭", bold=True, color=RGBColor(255, 255, 255), size=10)
    for c in hdr:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E79")
        c._tc.get_or_add_tcPr().append(shading)
    rows = [
        ("璁粌鏂瑰紡", "rolling / quarterly expanding / 2y"),
        ("棰勬祴鏍囩", "executable_5d_open_return"),
        ("鑲＄エ姹?, "all_a_light"),
        ("鎵ц鏂瑰紡", "娆℃棩寮€鐩樹环闄愪环涔板叆锛屽埌鏈熸棩寮€鐩樹环闄愪环鍗栧嚭锛屽紑鐩樻定鍋滆烦杩囦拱鍏ワ紝寮€鐩樿穼鍋滈『寤跺崠鍑?),
        ("鍥炴祴鍙ｅ緞", "鎵嬬画璐?0.03% + 婊戠偣 0.10%锛屽畼鏂?clean 鍥炴祴鏃ュ織 0 WARN"),
        ("淇″彿鍛ㄦ湡", f"{strategy['first_buy']} 鑷?{strategy['last_buy']}"),
        ("鏃ュ織鏂囦欢", strategy["log_file"]),
        ("淇″彿鏂囦欢", strategy["signal_file"]),
    ]
    for left, right in rows:
        cells = table.add_row().cells
        set_cell_text(cells[0], left, bold=True)
        set_cell_text(cells[1], right)

    add_heading(doc, "瑙勫垯鎷嗚В", level=2)
    add_bullets(doc, strategy["strategy_shape"])


def build_docx() -> Path:
    charts = build_charts()
    doc = Document()
    style_doc(doc)
    add_title_block(doc)

    add_heading(doc, "鏍稿績缁撹", level=1)
    add_bullets(
        doc,
        [
            "鏀剁泭鐜囨渶楂樼増鏇村亸杩涙敾锛屾牳蹇冨湪浜庝笉鍋氬競鍦鸿繃婊わ紝骞舵妸鍗曠瑪浠撲綅鎶埌 0.33锛屼粠鑰屾妸鎬绘毚闇叉帹鍒版帴杩戞弧浠撴粴鍔ㄣ€?,
            "澶忔櫘鐗堝姞浠撳悗鏇村亸绋冲仴锛屾牳蹇冨湪浜庝繚鐣欌€滃弻鎸囨暟閮界珯涓?MA20 鎵嶅紑浠撯€濈殑甯傚満鐜闂搁棬锛屽悓鏃舵妸鍗曠瑪浠撲綅涔熸姮鍒?0.33銆?,
            "涓ょ増閮介噰鐢ㄧ浉鍚岀殑 clean 鎵ц閾撅細鏄惧紡寮€鐩樹环闄愪环鍗曘€佸紑鐩樻定鍋滆烦杩囥€佸紑鐩樿穼鍋滈『寤躲€佸畼鏂瑰洖娴嬫棩蹇?0 WARN銆?,
        ],
    )

    add_heading(doc, "涓€椤靛姣?, level=1)
    comparison = doc.add_table(rows=1, cols=3)
    comparison.style = "Table Grid"
    set_table_borders(comparison)
    headers = comparison.rows[0].cells
    for idx, text in enumerate(["鎸囨爣", "鏀剁泭鐜囨渶楂樼増", "澶忔櫘鐗堝姞浠撳悗"]):
        set_cell_text(headers[idx], text, bold=True, color=RGBColor(255, 255, 255), size=10)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "1F4E79")
        headers[idx]._tc.get_or_add_tcPr().append(shading)
    metrics_rows = [
        ("骞村寲鏀剁泭鐜?, "203.50%", "130.52%"),
        ("绱鏀剁泭鐜?, "410.91%", "221.35%"),
        ("澶忔櫘姣旂巼", "2.0598", "2.7345"),
        ("鏈€澶у洖鎾?, "14.39%", "7.46%"),
        ("骞冲潎鎸佷粨鐜?, "85.55%", "56.49%"),
        ("宄板€兼寔浠撶巼", "99%", "99%"),
        ("寮€浠撴鏁?, "353", "174"),
        ("鑳滅巼", "49.29%", "55.17%"),
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

    add_heading(doc, "鎬庝箞閫?, level=1)
    add_bullets(
        doc,
        [
            "濡傛灉鐩爣鏄敖閲忔妸鏀剁泭鐜囧仛楂橈紝鑰屼笖鎺ュ彈鏇村ぇ鐨勮祫閲戞毚闇蹭笌鏇撮珮鐨勫洖鎾ゆ尝鍔紝灏遍€夋敹鐩婄巼鏈€楂樼増銆?,
            "濡傛灉鐩爣鏄鏀剁泭鏇茬嚎鏇寸ǔ銆佸洖鎾ゆ洿浣庛€侀闄╄皟鏁村悗鏀剁泭鏇存紓浜紝灏遍€夊鏅増鍔犱粨鍚庛€?,
            "鏀剁泭鐗堟洿鍍忊€滄寔缁繘鏀烩€濓紝澶忔櫘鐗堟洿鍍忊€滃甫甯傚満鐜闂搁棬鐨勭ǔ鍋ヨ繘鏀烩€濄€?,
        ],
    )

    add_heading(doc, "鏉ユ簮鏉愭枡", level=1)
    add_bullets(
        doc,
        [
            r"鍙傝€?PDF锛欴:\download\edge\婊′粨鏀剁泭鐗?pdf",
            r"鍙傝€?PDF锛欴:\download\edge\澶忔櫘.pdf",
            "璇存槑锛氳繖浠借鏄庝功閲嶆柊缁熶竴浜嗘寚鏍囧彛寰勩€佹墽琛岃鍒欏拰鍥捐〃琛ㄧ幇锛屽洜姝ゆ渶缁堟暟瀛椾互鏈鏄庝功鍒楀嚭鐨勫畼鏂?clean 鏃ュ織涓哄噯銆?,
        ],
    )

    doc.save(OUTPUT_DOCX)
    return OUTPUT_DOCX


if __name__ == "__main__":
    path = build_docx()
    print(path)

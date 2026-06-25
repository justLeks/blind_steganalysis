from __future__ import annotations

import csv
import copy
import math
import os
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


# Paper-production tooling: edits a reviewer's DOCX. Point THESES_SOURCE_DOCX at your local copy
# (defaults to a file of this name in the current directory) -- no machine-specific path is committed.
SOURCE_DOCX = Path(
    os.environ.get("THESES_SOURCE_DOCX", "texture_energy_probing_conference_theses_en_revised_d_progonov.docx")
)
SUMMARY_CSV = Path("experiments/probing_only/probe_summary.csv")
OUTPUT_DOCX = Path("texture_energy_probing_conference_theses_en_revised_green.docx")
ASSET_DIR = Path("generated_thesis_assets")
HUGO_CHART = ASSET_DIR / "hugo_texture_based_guessed_pixels_semilog.png"
MIPOD_CHART = ASSET_DIR / "mipod_texture_based_guessed_pixels_semilog.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NSMAP = {
    "wpc": "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    "mo": "http://schemas.microsoft.com/office/mac/office/2008/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "o": "urn:schemas-microsoft-com:office:office",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v": "urn:schemas-microsoft-com:vml",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "w10": "urn:schemas-microsoft-com:office:word",
    "w": W_NS,
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "wpi": "http://schemas.microsoft.com/office/word/2010/wordprocessingInk",
    "wne": "http://schemas.microsoft.com/office/word/2006/wordml",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
}


def wtag(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def read_summary(path: Path = SUMMARY_CSV) -> dict[tuple[str, float, float], dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["method"].upper(), float(row["alpha"]), float(row["probe_budget_percentage"])): row
        for row in rows
    }


def metric(lookup: dict[tuple[str, float, float], dict], method: str, alpha: float, budget: float, key: str) -> float:
    return float(lookup[(method.upper(), alpha, budget)][key])


def fmt_pixels(value: float) -> str:
    return str(int(round(value)))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def alpha_pct(alpha: float) -> str:
    return f"{int(round(alpha * 100))}%"


def alpha_label(alpha: float) -> str:
    return alpha_pct(alpha)


def fit_power_model(lookup: dict[tuple[str, float, float], dict], method: str, alpha: float) -> tuple[float, float, float]:
    budgets = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    x_values = [math.log(value) for value in budgets]
    y_values = [
        math.log(metric(lookup, method, alpha, budget, "mean_guessed_carrier_pixels"))
        for budget in budgets
    ]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    exponent = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / sum((x_value - x_mean) ** 2 for x_value in x_values)
    intercept = y_mean - exponent * x_mean
    coefficient = math.exp(intercept)
    predictions = [intercept + exponent * value for value in x_values]
    residual_sum = sum((value - prediction) ** 2 for value, prediction in zip(y_values, predictions))
    total_sum = sum((value - y_mean) ** 2 for value in y_values)
    r_squared = 1.0 - residual_sum / total_sum
    return coefficient, exponent, r_squared


def run_props(bold: bool = False, italic: bool = False, size: int = 24, green: bool = False, subscript: bool = False) -> str:
    bold_xml = "<w:b/><w:bCs/>" if bold else ""
    italic_xml = "<w:i/><w:iCs/>" if italic else ""
    color_xml = '<w:color w:val="008000"/>' if green else ""
    subscript_xml = '<w:vertAlign w:val="subscript"/>' if subscript else ""
    return (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
        f"{bold_xml}{italic_xml}{color_xml}{subscript_xml}"
        f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
        "</w:rPr>"
    )


def text_run(
    text: str,
    bold: bool = False,
    italic: bool = False,
    size: int = 24,
    green: bool = False,
    subscript: bool = False,
) -> str:
    space = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return (
        f"<w:r>{run_props(bold=bold, italic=italic, size=size, green=green, subscript=subscript)}"
        f"<w:t{space}>{escape(text)}</w:t></w:r>"
    )


def styled_text_runs(text: str, *, bold: bool = False, italic: bool = False, size: int = 24, green: bool = False) -> str:
    pieces = re.split(r"([A-Za-zΑ-Ωα-ω]+_[A-Za-zΑ-Ωα-ω0-9]+)", text)
    out: list[str] = []
    for piece in pieces:
        if "_" in piece and re.fullmatch(r"[A-Za-zΑ-Ωα-ω]+_[A-Za-zΑ-Ωα-ω0-9]+", piece):
            base, subscript = piece.split("_", maxsplit=1)
            out.append(text_run(base, bold=bold, italic=italic, size=size, green=green))
            out.append(text_run(subscript, bold=bold, italic=italic, size=size - 4, green=green, subscript=True))
        elif piece:
            out.append(text_run(piece, bold=bold, italic=italic, size=size, green=green))
    return "".join(out)


def paragraph(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    size: int = 24,
    before: int = 60,
    after: int = 60,
    align: str | None = "both",
    green: bool = False,
) -> str:
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    return (
        "<w:p>"
        f'<w:pPr><w:spacing w:before="{before}" w:after="{after}"/>{jc}</w:pPr>'
        f"{styled_text_runs(text, bold=bold, italic=italic, size=size, green=green)}"
        "</w:p>"
    )


def title(text: str, *, green: bool = False) -> str:
    return paragraph(text, bold=True, size=30, before=80, after=120, align="center", green=green)


def heading(text: str, *, green: bool = False) -> str:
    return paragraph(text, bold=True, size=26, before=180, after=80, align=None, green=green)


def caption(text: str, *, green: bool = False) -> str:
    return paragraph(text, bold=True, size=20, before=120, after=40, align="center", green=green)


def table_cell(text: str, width: int, *, header: bool = False, center: bool = False, green: bool = False) -> str:
    jc = "center" if center or header else "both"
    shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9EAF7"/>' if header else ""
    return (
        "<w:tc>"
        f'<w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
        "<w:p>"
        f'<w:pPr><w:spacing w:before="20" w:after="20"/><w:jc w:val="{jc}"/></w:pPr>'
        f"{styled_text_runs(text, bold=header, size=20, green=green)}"
        "</w:p>"
        "</w:tc>"
    )


def table(rows: list[list[str]], widths: list[int], center_cols: set[int] | None = None, *, green: bool = False) -> str:
    center_cols = center_cols or set()
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    out = [
        "<w:tbl>",
        "<w:tblPr>",
        '<w:tblW w:w="0" w:type="auto"/>',
        '<w:tblInd w:w="10" w:type="dxa"/>',
        "<w:tblBorders>",
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        '<w:left w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        '<w:right w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        '<w:insideH w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        '<w:insideV w:val="single" w:sz="6" w:space="0" w:color="444444"/>',
        "</w:tblBorders>",
        '<w:tblCellMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>',
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>',
        "</w:tblPr>",
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    for row_index, row in enumerate(rows):
        out.append("<w:tr>")
        for col_index, value in enumerate(row):
            out.append(
                table_cell(
                    value,
                    widths[col_index],
                    header=row_index == 0,
                    center=col_index in center_cols,
                    green=green,
                )
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def build_terms_table() -> list[list[str]]:
    return [
        ["Notation", "Meaning in this work"],
        ["C", "Set of carrier pixels: positions modified by simulated conseal embedding."],
        ["P(B)", "Set of unique positions inspected at probing budget B."],
        ["H(B)", "Correctly identified carrier pixels: |P(B) ∩ C|."],
        ["Cov(B)", "Carrier coverage: H(B) / |C|."],
        ["Hit(B)", "Probe hit fraction: H(B) / |P(B)|."],
    ]


def build_results_table(lookup: dict[tuple[str, float, float], dict]) -> list[list[str]]:
    rows = [
        [
            "Embedding level, δ_α",
            "HUGO carriers",
            "HUGO guessed, 1%",
            "HUGO guessed, 10%",
            "MiPOD carriers",
            "MiPOD guessed, 1%",
            "MiPOD guessed, 10%",
        ]
    ]
    for alpha in [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
        rows.append(
            [
                alpha_pct(alpha),
                fmt_pixels(metric(lookup, "HUGO", alpha, 1.0, "mean_carrier_pixels")),
                fmt_pixels(metric(lookup, "HUGO", alpha, 1.0, "mean_guessed_carrier_pixels")),
                fmt_pixels(metric(lookup, "HUGO", alpha, 10.0, "mean_guessed_carrier_pixels")),
                fmt_pixels(metric(lookup, "MIPOD", alpha, 1.0, "mean_carrier_pixels")),
                fmt_pixels(metric(lookup, "MIPOD", alpha, 1.0, "mean_guessed_carrier_pixels")),
                fmt_pixels(metric(lookup, "MIPOD", alpha, 10.0, "mean_guessed_carrier_pixels")),
            ]
        )
    return rows


def build_precision_table(lookup: dict[tuple[str, float, float], dict]) -> list[list[str]]:
    rows = [["Method", "Embedding level, δ_α", "1% budget", "10% budget", "50% budget"]]
    for method in ["HUGO", "MIPOD"]:
        for alpha in [0.03, 0.1, 0.3, 0.5]:
            rows.append(
                [
                    method_label(method),
                    alpha_pct(alpha),
                    fmt_pct(metric(lookup, method, alpha, 1.0, "mean_precision_hit_rate")),
                    fmt_pct(metric(lookup, method, alpha, 10.0, "mean_precision_hit_rate")),
                    fmt_pct(metric(lookup, method, alpha, 50.0, "mean_precision_hit_rate")),
                ]
            )
    return rows


def build_power_model_table(lookup: dict[tuple[str, float, float], dict]) -> list[list[str]]:
    rows = [["Method", "Embedding level, δ_α", "Coefficient a", "Exponent p", "R²"]]
    for method in ["HUGO", "MIPOD"]:
        for alpha in [0.03, 0.1, 0.3, 0.5]:
            coefficient, exponent, r_squared = fit_power_model(lookup, method, alpha)
            rows.append([method_label(method), alpha_pct(alpha), f"{coefficient:.2f}", f"{exponent:.3f}", f"{r_squared:.4f}"])
    return rows


def method_label(method: str) -> str:
    return "MiPOD" if method.upper() == "MIPOD" else method.upper()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rounded_axis_max(value: float) -> int:
    if value <= 1000:
        step = 100
    elif value <= 5000:
        step = 500
    else:
        step = 1000
    return int(((value + step - 1) // step) * step)


def draw_pattern_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    pattern: tuple[int, int] | None,
) -> None:
    if pattern is None:
        draw.line(points, fill=fill, width=width)
        return
    dash, gap = pattern
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        position = 0.0
        while position < distance:
            draw_end = min(distance, position + dash)
            x1 = start[0] + dx * position / distance
            y1 = start[1] + dy * position / distance
            x2 = start[0] + dx * draw_end / distance
            y2 = start[1] + dy * draw_end / distance
            draw.line((x1, y1, x2, y2), fill=fill, width=width)
            position += dash + gap


def draw_line_chart(
    lookup: dict[tuple[str, float, float], dict],
    method: str,
    output_path: Path,
) -> None:
    budgets = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    alphas = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    series = {
        alpha: [metric(lookup, method, alpha, budget, "mean_guessed_carrier_pixels") for budget in budgets]
        for alpha in alphas
    }
    fit_params = {
        alpha: fit_power_model(lookup, method, alpha)
        for alpha in alphas
    }
    fit_x_values = [1.0 + index * 0.5 for index in range(99)]
    fit_series = {
        alpha: [fit_params[alpha][0] * (budget ** fit_params[alpha][1]) for budget in fit_x_values]
        for alpha in alphas
    }
    width, height = 1600, 900
    plot_left, plot_top, plot_right, plot_bottom = 140, 120, 1190, 750
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    label_font = load_font(28)
    small_font = load_font(24)
    legend_font = load_font(24)
    colors = ["#151515", "#333333", "#4b4b4b", "#626262", "#777777", "#909090", "#aaaaaa"]
    patterns = [None, (18, 10), (5, 7), (28, 10), (12, 6), (3, 5), (36, 8)]
    log_min = 0.0
    max_value = max(
        max(max(values) for values in series.values()),
        max(max(values) for values in fit_series.values()),
    )
    log_max = math.ceil(math.log10(max_value))
    max_y = 10 ** log_max

    draw.text((80, 35), f"{method_label(method)}: correctly identified carrier pixels", fill="black", font=title_font)
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#333333", width=2)

    for exponent in range(int(log_min), log_max + 1):
        y_value = 10 ** exponent
        y = plot_bottom - (plot_bottom - plot_top) * (math.log10(y_value) - log_min) / (log_max - log_min)
        draw.line((plot_left, y, plot_right, y), fill="#dddddd", width=1)
        draw.text((35, y - 14), str(y_value), fill="black", font=small_font)

    for budget in budgets:
        x = plot_left + (plot_right - plot_left) * (budget - budgets[0]) / (budgets[-1] - budgets[0])
        draw.line((x, plot_bottom, x, plot_bottom + 8), fill="#333333", width=2)
        draw.text((x - 20, plot_bottom + 18), f"{int(budget)}%", fill="black", font=small_font)

    draw.text((plot_left + 360, height - 80), "Probing budget", fill="black", font=label_font)
    draw.text((18, 65), "pixels (log scale)", fill="black", font=small_font)

    for color, pattern, alpha in zip(colors, patterns, alphas):
        fit_points = []
        for budget, y_value in zip(fit_x_values, fit_series[alpha]):
            x = plot_left + (plot_right - plot_left) * (budget - budgets[0]) / (budgets[-1] - budgets[0])
            y = plot_bottom - (plot_bottom - plot_top) * (math.log10(y_value) - log_min) / (log_max - log_min)
            fit_points.append((x, y))
        draw_pattern_line(draw, fit_points, color, 2, (8, 8))

        points = []
        for budget, y_value in zip(budgets, series[alpha]):
            x = plot_left + (plot_right - plot_left) * (budget - budgets[0]) / (budgets[-1] - budgets[0])
            y = plot_bottom - (plot_bottom - plot_top) * (math.log10(y_value) - log_min) / (log_max - log_min)
            points.append((x, y))
        draw_pattern_line(draw, points, color, 5, pattern)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)

    legend_x, legend_y = 1230, 150
    draw.text((legend_x, legend_y - 50), "Embedding level", fill="black", font=label_font)
    for idx, (color, pattern, alpha) in enumerate(zip(colors, patterns, alphas)):
        y = legend_y + idx * 55
        draw_pattern_line(draw, [(legend_x, y + 12), (legend_x + 55, y + 12)], color, 5, pattern)
        draw.ellipse((legend_x + 22, y + 5, legend_x + 36, y + 19), fill=color, outline="white", width=2)
        draw.text((legend_x + 75, y), alpha_label(alpha), fill="black", font=legend_font)

    params_y = 555
    params_font = load_font(18)
    params_title_font = load_font(20, bold=True)
    draw.text((legend_x, params_y), "Power fit H=aB^p", fill="black", font=params_title_font)
    draw.text((legend_x, params_y + 30), "level     a      p     R²", fill="black", font=params_font)
    for idx, alpha in enumerate(alphas):
        coefficient, exponent, r_squared = fit_params[alpha]
        y = params_y + 58 + idx * 32
        draw.text(
            (legend_x, y),
            f"{alpha_label(alpha):>3}  {coefficient:6.1f}  {exponent:.3f}  {r_squared:.3f}",
            fill="black",
            font=params_font,
        )
    draw_pattern_line(draw, [(legend_x, params_y + 300), (legend_x + 55, params_y + 300)], "#555555", 2, (8, 8))
    draw.text((legend_x + 75, params_y + 287), "fitted curve", fill="black", font=params_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def build_chart_images(lookup: dict[tuple[str, float, float], dict]) -> dict[str, Path]:
    draw_line_chart(lookup, "HUGO", HUGO_CHART)
    draw_line_chart(lookup, "MIPOD", MIPOD_CHART)
    return {"rId20": HUGO_CHART, "rId21": MIPOD_CHART}


def image_paragraph(rel_id: str, name: str, width_emu: int = 5669280, height_emu: int = 3194550) -> str:
    return f"""
<w:p>
  <w:pPr><w:spacing w:before="80" w:after="80"/><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{width_emu}" cy="{height_emu}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{20 if rel_id == 'rId20' else 21}" name="{escape(name)}"/>
        <wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>
        <a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:nvPicPr><pic:cNvPr id="0" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def document_body(
    sect_pr: ET.Element,
    lookup: dict[tuple[str, float, float], dict],
) -> str:
    sect_pr_xml = ET.tostring(sect_pr, encoding="unicode")
    parts: list[str] = []
    parts.append(title("A TEXTURE-ORIENTED METHOD FOR PROBING CARRIER PIXELS IN HUGO AND MIPOD STEGO IMAGES", green=True))
    parts.append(heading("Abstract"))
    parts.append(
        paragraph(
            "This paper proposes a texture-oriented probabilistic method for probing carrier pixels in stego images. The method assigns probing probabilities from local gray-level variation and then inspects unique pixel positions under a limited budget. The evaluation uses 100 grayscale 512 x 512 ALASKA images and simulated embedding by the HUGO and MiPOD implementations in conseal. The studied embedding levels are δ_α = 3%, 5%, 10%, 20%, 30%, 40%, and 50%. At a 1% probing budget and δ_α=3%, the method identifies 43 HUGO carrier pixels and 14 MiPOD carrier pixels. At δ_α=50%, the corresponding numbers are 538 and 406. The dependence of correctly identified carrier pixels on budget is approximated by a power model H(B)=aB^p, with R² above 0.988 for all studied settings. The results show that texture-oriented probing provides useful spatial information and responds differently to the two embedding algorithms.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Keywords: image steganalysis, texture-oriented probing, carrier pixel, HUGO, MiPOD, ALASKA, selection channel.",
            bold=True,
            align="both",
            green=True,
        )
    )

    parts.append(heading("Introduction"))
    parts.append(
        paragraph(
            "Adaptive image steganography attempts to conceal data by modifying pixels whose changes are less statistically detectable. In the prisoners' problem, such modifications support a covert communication channel inside an apparently ordinary digital object [1]. Passive steganalysis primarily determines whether hidden data are present, whereas a spatial probing task asks which image positions are likely to have carried embedding changes. The latter problem is relevant when only a limited number of pixel positions can be inspected.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Existing residual-based and selection-channel-aware approaches provide features for detection. Payload-location studies have also located modifications under restrictive assumptions, such as repeated LSB embedding paths across many stego images [7]. These approaches do not directly provide a transparent budgeted rule for inspecting one adaptively embedded stego image. The gap addressed here is therefore an image-adaptive sampling rule whose correctly identified carrier positions can be counted as the probing budget and embedding level change.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The aim of the work is to propose and evaluate a texture-oriented probabilistic method for identifying carrier-pixel positions under fixed probing budgets. Three objectives are addressed. First, the texture-oriented probability map and evaluation measures are formalised. Second, HUGO and MiPOD stego images are evaluated at several embedding levels. Third, the dependence between probing budget and the number of correctly identified carrier pixels is analysed.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The next section positions this objective relative to existing detection and payload-location studies before the proposed probing rule is defined.",
            green=True,
        )
    )

    parts.append(heading("1. Related work and motivation", green=True))
    parts.append(
        paragraph(
            "Rich-model steganalysis is based on high-dimensional residual statistics designed to capture weak local disturbances introduced by embedding [2]. Selection-channel-aware rich models extend this idea by including information about unequal probabilities of modifying individual pixels [3]. Residual representations therefore describe local statistical evidence, while a selection channel describes where adaptive embedding is more likely to operate. In this work, probing uses an estimated spatial preference only to select candidate pixels; it is not a stego-image detector or a recovered selection channel.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Payload location is the closest established task to the present one. Ker located LSB-replacement payload via weighted stego-image residuals when multiple images reused identical embedding positions [7]. The present setting differs in two respects: HUGO and MiPOD are content-adaptive algorithms, and each stego image is probed independently. The term probing is used operationally here for budgeted sampling of candidate positions followed by counting intersections with a known experimental carrier mask.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The proposed texture-oriented method combines first-order intensity differences with a four-neighbour residual. These components define one content-dependent probability map for selecting candidate carrier pixels. Its contribution is a formally specified budgeted probing procedure and an analysis of how carrier identification changes with budget, embedding level, and embedding algorithm.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "These distinctions determine the hypothesis and the pixel-scoring rule formalised in the following section.",
            green=True,
        )
    )

    parts.append(heading("2. Proposed texture-oriented probing method", green=True))
    parts.append(
        paragraph(
            "Research hypothesis. If adaptive embedding assigns lower modification costs to locally irregular areas, the carrier-identification response of the proposed texture-oriented map will depend systematically on the probing budget and embedding level. The response is analysed separately for HUGO and MiPOD because their embedding objectives are different.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "In this paper, texture does not denote the semantic texture of an object such as grass, fabric, or a wall. It denotes pixel-scale spatial variation computed independently from every analysed stego image. The name texture-oriented is used because the score favours locally varying regions; it is not asserted to be a general physical energy measure.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Let I(i,j) be the luminance at a pixel position. The local components are the horizontal difference D_h, the vertical difference D_v, and the absolute four-neighbour residual R_4:",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "D_h(i,j)=|I(i,j)-I(i,j-1)|,    D_v(i,j)=|I(i,j)-I(i-1,j)|,",
            italic=True,
            before=40,
            after=40,
            align="center",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "R_4(i,j)=|4I(i,j)-I(i-1,j)-I(i+1,j)-I(i,j-1)-I(i,j+1)|.",
            italic=True,
            before=40,
            after=40,
            align="center",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The proposed score is E(i,j)=D_h(i,j)+D_v(i,j)+R_4(i,j). Large values correspond to edges or local gray-level oscillations. They are used as evidence of regions in which an adaptive embedding algorithm may tolerate changes at lower statistical cost.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The probing probability is obtained by normalising this score. In the implementation, ε=10⁻⁶ keeps flat pixels selectable, while γ=1 preserves a linear relation between the score and sampling weight:",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "p(i,j) = (E(i,j)+ε)^γ / Σ (E(u,v)+ε)^γ, for all (u,v);    ε=10⁻⁶, γ=1.",
            italic=True,
            before=40,
            after=40,
            align="center",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Let C be the recorded set of carrier pixels and let P(B) be the set of unique probed positions for budget B. Sampling is performed without replacement. One weighted random order is generated up to the maximum budget. Therefore, P(1%) is contained in P(5%), P(5%) is contained in P(10%), and so forth. This nesting shows how extra budget extends the same probing path.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The notation and evaluation measures used below are summarised in Table 1. In particular, H(B) is a direct count rather than a classifier statistic; Cov(B) and Hit(B) are normalised auxiliary measures derived from the known carrier mask.",
            green=True,
        )
    )
    parts.append(caption("Table 1. Notation and measures of the proposed probing method", green=True))
    parts.append(table(build_terms_table(), [2100, 7200], center_cols={0}, green=True))
    parts.append(
        paragraph(
            "Table 1 separates the primary count H(B) from two normalised descriptions. Cov(B) answers what share of all modified positions is recovered, whereas Hit(B) answers what share of inspected positions is correct. This distinction prevents a larger budget from being misinterpreted as a more selective probe.",
            green=True,
        )
    )

    parts.append(heading("3. Experimental methodology", green=True))
    parts.append(
        paragraph(
            "The evaluation uses 100 grayscale 512 x 512 images from the ALASKA dataset [6]. Stego images are generated by the HUGO and MiPOD single-channel simulators provided by conseal [8]. Here simulated embedding means that modification positions are sampled at a prescribed payload; no secret-message encoder or decoder is evaluated. For each stego image, the difference from its cover image records the experimental carrier set C.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The embedding parameter α is reported as embedding level δ_α=100α%. The analysed levels are 3%, 5%, 10%, 20%, 30%, 40%, and 50%. A configuration is defined by an image, embedding algorithm, δ_α, texture-oriented probability map, probing budget, and seed. The budgets B = 1%, 5%, 10%, 20%, 30%, 40%, and 50% refer to all pixels of an input image. Thus, B=1% corresponds to 2,621 unique positions.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "For the reported experiment, the embedding seed is fixed at 12345 and the probing seed is fixed at 41. Image-specific simulator seeds are derived reproducibly from the embedding seed and image path. The stored seeds reproduce the same stego images and the same texture-oriented weighted probe order.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The algorithms favour modifications by different objective principles. For HUGO, a payload-constrained distortion principle is represented by the first expression below [4]. MiPOD uses the second principle, in which detectability under a statistical cover model is minimised for a fixed payload [5]:",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "HUGO: min_π Σ_i ρ_i π_i, subject to Payload(π)=m;    MiPOD: min_β D(β), subject to Payload(β)=m.",
            italic=True,
            before=40,
            after=40,
            align="center",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "Here ρ_i is the HUGO modification cost for position i, π_i is its change probability, D(β) is the MiPOD detectability criterion, and m is the requested payload. Conseal implements the corresponding simulators [8]. The present work observes the changed positions generated by those simulators; it does not reimplement either objective.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The main outcome is H(B), the mean number of correctly identified carrier pixels across 100 images for each algorithm and embedding level δ_α. The auxiliary measures are Cov(B)=H(B)/|C| and Hit(B)=H(B)/|P(B)|. The reported values describe one reproducible seeded run of the proposed method; inferential claims would require repeated seeds and confidence intervals.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The resulting carrier counts, curves, and fitted growth laws are presented and interpreted in the next section.",
            green=True,
        )
    )

    parts.append(heading("4. Results and discussion", green=True))
    parts.append(
        paragraph(
            "Table 2 provides the direct carrier-count outcome of texture-oriented probing. As expected, larger embedding levels δ_α produce more carrier pixels and consequently larger H(B) values. More importantly, the counts permit the functional dependence on the probing budget to be analysed rather than merely reporting an increasing trend.",
            green=True,
        )
    )
    parts.append(caption("Table 2. Mean correctly identified carrier pixels H(B) for texture-oriented probing", green=True))
    parts.append(table(build_results_table(lookup), [850, 1400, 1400, 1450, 1400, 1400, 1400], center_cols={0, 1, 2, 3, 4, 5, 6}, green=True))
    parts.append(
        paragraph(
            "For HUGO stego images, at B=1% the method identifies 43 carrier pixels at embedding level δ_α=3% and 538 at δ_α=50%. At B=10%, it identifies 363 and 5153 pixels, respectively. For MiPOD images, the corresponding B=1% counts are 14 and 406. At B=10%, they are 134 and 3953.",
            green=True,
        )
    )
    parts.append(caption("Figure 1. HUGO stego images: correctly identified carrier pixels on a logarithmic ordinate", green=True))
    parts.append(image_paragraph("rId20", "HUGO texture-oriented probing graph"))
    parts.append(
        paragraph(
            "Figure 1 uses a logarithmic ordinate and line patterns suitable for grayscale printing. The approximately straight trajectories indicate a power-type response of H(B) to B. The HUGO curves spread more noticeably between low and high embedding levels δ_α. This spread indicates stronger variation of the fitted exponent.",
            green=True,
        )
    )
    parts.append(caption("Figure 2. MiPOD stego images: correctly identified carrier pixels on a logarithmic ordinate", green=True))
    parts.append(image_paragraph("rId21", "MiPOD texture-oriented probing graph"))
    parts.append(
        paragraph(
            "Figure 2 displays a similar power-type relation for MiPOD stego images, but the trajectories are more nearly parallel. This observation suggests that the response exponent changes less with embedding level δ_α for MiPOD than for HUGO.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "To quantify this observation, Table 3 reports the fitted power model H(B)=aB^p. For HUGO, the exponent increases from 0.806 at embedding level δ_α=3% to 0.935 at δ_α=50%. For MiPOD, it changes within a narrower range, from 0.935 to 0.959. All displayed fits achieve R² above 0.988. The nontrivial result is not merely that additional budget finds more carriers. The growth law and its sensitivity to embedding level differ between the two embedding algorithms.",
            green=True,
        )
    )
    parts.append(caption("Table 3. Power-model parameters for H(B)=aB^p", green=True))
    parts.append(table(build_power_model_table(lookup), [1500, 1700, 2100, 2000, 2000], center_cols={0, 1, 2, 3, 4}, green=True))
    parts.append(
        paragraph(
            "Table 3 confirms the visual pattern in Figures 1 and 2. The exponent p increases more strongly across HUGO embedding levels than across MiPOD embedding levels. The high R² values support using the power model as a compact descriptive summary of texture-oriented probing.",
            green=True,
        )
    )

    parts.append(heading("Conclusions", green=True))
    parts.append(
        paragraph(
            "A texture-oriented probabilistic method for budgeted carrier-pixel probing has been proposed and formally specified. It converts local first-order differences and a four-neighbour residual into a sampling distribution. The method evaluates H(B), the number of correctly identified carrier positions. For a 1% budget and embedding level δ_α=3%, it identifies 43 HUGO carrier pixels and 14 MiPOD carrier pixels. At δ_α=50%, it identifies 538 and 406 pixels, respectively.",
            green=True,
        )
    )
    parts.append(
        paragraph(
            "The principal analytical result is the observed power-type dependence H(B)=aB^p. The exponent changes substantially with embedding level for HUGO and less strongly for MiPOD. This behaviour demonstrates different correspondence between the proposed texture-oriented map and the two carrier-selection mechanisms. Further validation should use repeated probing seeds, confidence intervals, and independent image samples.",
            green=True,
        )
    )

    parts.append(heading("References"))
    references = [
        '[1] G. J. Simmons, "The prisoners\' problem and the subliminal channel," CRYPTO, 1983.',
        '[2] J. Fridrich and J. Kodovsky, "Rich models for steganalysis of digital images," IEEE TIFS, vol. 7, no. 3, 2012.',
        '[3] T. Denemark, J. Fridrich, and V. Holub, "Selection-channel-aware rich model for steganalysis of digital images," IEEE WIFS, 2014.',
        '[4] T. Pevny, T. Filler, and P. Bas, "Using high-dimensional image models to perform highly undetectable steganography," Information Hiding, 2010.',
        '[5] V. Sedighi, R. Cogranne, and J. Fridrich, "Content-adaptive steganography by minimizing statistical detectability," IEEE TIFS, vol. 11, no. 2, pp. 221-234, 2016.',
        '[6] Q. Giboulot, R. Cogranne, and P. Bas, "ALASKA#2: challenging academic research on steganalysis with realistic images," IEEE WIFS, 2020.',
        '[7] A. D. Ker, "Locating steganographic payload via WS residuals," in Proc. ACM Multimedia and Security Workshop, pp. 27-32, 2008.',
        '[8] DDE Laboratory, "conseal 2024.12 documentation: simulators of spatial steganography," [Online]. Available: https://conseal.readthedocs.io/en/latest/reference.html. Accessed: May 25, 2026.',
    ]
    for ref in references:
        parts.append(paragraph(ref, size=22, before=20, after=20, align="both"))

    return "<w:body>" + "".join(parts) + sect_pr_xml + "</w:body>"


def build_document_xml(
    original_xml: bytes,
    lookup: dict[tuple[str, float, float], dict],
) -> bytes:
    root = ET.fromstring(original_xml)
    body = root.find(wtag("body"))
    if body is None:
        raise ValueError("Source DOCX has no document body.")
    sect_pr = body.find(wtag("sectPr"))
    if sect_pr is None:
        raise ValueError("Source DOCX has no section properties.")

    ns_attrs = " ".join(f'xmlns:{prefix}="{uri}"' for prefix, uri in NSMAP.items())
    body_xml = document_body(copy.deepcopy(sect_pr), lookup)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document {ns_attrs} mc:Ignorable="w14 wp14">'
        f"{body_xml}"
        "</w:document>"
    )
    return document_xml.encode("utf-8")


def build_document_rels_xml(original_rels_xml: bytes) -> bytes:
    rels_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    image_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    chart_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"
    root = ET.fromstring(original_rels_xml)
    for rel in list(root):
        rel_type = rel.get("Type", "")
        if rel_type in {image_type, chart_type} or "comments" in rel_type.lower() or rel_type.endswith("/people"):
            root.remove(rel)
    ET.SubElement(
        root,
        f"{{{rels_ns}}}Relationship",
        {
            "Id": "rId20",
            "Type": image_type,
            "Target": "media/hugo_texture_based_guessed_pixels_semilog.png",
        },
    )
    ET.SubElement(
        root,
        f"{{{rels_ns}}}Relationship",
        {
            "Id": "rId21",
            "Type": image_type,
            "Target": "media/mipod_texture_based_guessed_pixels_semilog.png",
        },
    )
    ET.register_namespace("", rels_ns)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def build_content_types_xml(original_content_types_xml: bytes) -> bytes:
    ct_ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    root = ET.fromstring(original_content_types_xml)
    for child in list(root):
        part_name = child.get("PartName", "")
        if (
            part_name.startswith("/word/charts/")
            or "comment" in part_name.lower()
            or part_name == "/word/people.xml"
        ):
            root.remove(child)
    ET.register_namespace("", ct_ns)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(
            f"Source DOCX not found: {SOURCE_DOCX}. Set THESES_SOURCE_DOCX to your local copy "
            "(this is paper-production tooling and needs the reviewer's .docx)."
        )
    lookup = read_summary(SUMMARY_CSV)
    build_chart_images(lookup)
    original_xml = zipfile.ZipFile(SOURCE_DOCX).read("word/document.xml")
    new_document_xml = build_document_xml(original_xml, lookup)
    new_rels_xml = build_document_rels_xml(
        zipfile.ZipFile(SOURCE_DOCX).read("word/_rels/document.xml.rels")
    )
    new_content_types_xml = build_content_types_xml(
        zipfile.ZipFile(SOURCE_DOCX).read("[Content_Types].xml")
    )

    with zipfile.ZipFile(SOURCE_DOCX, "r") as source, zipfile.ZipFile(
        OUTPUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED
    ) as output:
        for info in source.infolist():
            if info.filename == "[Content_Types].xml":
                output.writestr(info, new_content_types_xml)
            elif info.filename == "word/document.xml":
                output.writestr(info, new_document_xml)
            elif info.filename == "word/_rels/document.xml.rels":
                output.writestr(info, new_rels_xml)
            elif (
                info.filename.startswith("word/charts/")
                or info.filename.startswith("word/media/")
                or info.filename.startswith("word/comments")
                or info.filename == "word/people.xml"
            ):
                continue
            else:
                output.writestr(info, source.read(info.filename))
        output.writestr("word/media/hugo_texture_based_guessed_pixels_semilog.png", HUGO_CHART.read_bytes())
        output.writestr("word/media/mipod_texture_based_guessed_pixels_semilog.png", MIPOD_CHART.read_bytes())

    print(f"Wrote {OUTPUT_DOCX.resolve()}")


if __name__ == "__main__":
    main()

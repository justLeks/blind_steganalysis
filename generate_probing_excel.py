from __future__ import annotations

import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import run_probing_experiment as experiment


# Edit these settings and run: python3 generate_probing_excel.py
EXPERIMENT_ROOT = experiment.EXPERIMENT_ROOT
RAW_CSV = experiment.RAW_CSV
SUMMARY_CSV = experiment.SUMMARY_CSV
OUTPUT_XLSX = EXPERIMENT_ROOT / "probing_results.xlsx"
METADATA_JSON = experiment.METADATA_JSON

METHODS = ["HUGO", "MIPOD"]
ALPHAS = experiment.ALPHAS
BUDGET_PERCENTAGES = [100.0 * fraction for fraction in experiment.PROBE_BUDGET_FRACTIONS]


@dataclass
class ChartSeries:
    name: str
    x_ref: str
    y_ref: str
    x_values: list[float]
    y_values: list[float]
    style_index: int | None = None
    is_fit: bool = False


@dataclass
class ChartSpec:
    sheet_name: str
    title: str
    series: list[ChartSeries]
    anchor_row: int
    anchor_col: int
    width_px: int = 720
    height_px: int = 360
    y_log: bool = False
    y_percent: bool = True
    x_title: str | None = None
    y_title: str | None = None
    grayscale_styles: bool = False


def xml_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_ref(row: int, col: int) -> str:
    return f"{column_name(col)}{row}"


def abs_ref(sheet_name: str, row: int, col: int) -> str:
    return f"'{sheet_name}'!${column_name(col)}${row}"


def abs_range(sheet_name: str, row1: int, col1: int, row2: int, col2: int) -> str:
    return f"{abs_ref(sheet_name, row1, col1)}:{abs_ref(sheet_name, row2, col2).split('!')[1]}"


def is_number(value: object) -> bool:
    if value is None:
        return False
    text = str(value)
    if text == "":
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number)


def as_number(value: object) -> float:
    return float(value)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def rows_from_dicts(headers: list[str], rows: list[dict]) -> list[list[object]]:
    return [headers] + [[row.get(header, "") for header in headers] for row in rows]


def method_label(method: str) -> str:
    return "MiPOD" if method.upper() == "MIPOD" else method.upper()


def alpha_label(alpha: float) -> str:
    return f"alpha={alpha:g}"


def cell_style(header: str | None, is_header: bool, value: object) -> int:
    if is_header:
        return 1
    if header is None:
        return 0
    if not is_number(value):
        return 0
    key = header.lower()
    if (
        "fraction" in key
        or "rate" in key
        or "recall" in key
        or "precision" in key
    ):
        return 2
    if "percentage" in key:
        return 4
    if "pixels" in key or key in {"records", "image_height", "image_width", "total_pixels"}:
        return 3
    return 0


def cell_xml(row: int, col: int, value: object, style: int = 0) -> str:
    ref = cell_ref(row, col)
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if is_number(value):
        return f'<c r="{ref}"{style_attr}><v>{float(value):.15g}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{xml_escape(value)}</t></is></c>'


def worksheet_xml(
    rows: list[list[object]],
    *,
    header_rows: set[int] | None = None,
    drawing_rid: str | None = None,
    freeze_top_row: bool = True,
    autofilter: bool = True,
    widths: list[float] | None = None,
) -> str:
    header_rows = header_rows or {1}
    headers = [str(value) for value in rows[0]] if rows else []
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
    ]
    if freeze_top_row:
        parts.append(
            '<sheetViews><sheetView workbookViewId="0">'
            '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
            '<selection pane="bottomLeft"/>'
            "</sheetView></sheetViews>"
        )
    if widths:
        parts.append("<cols>")
        for index, width in enumerate(widths, start=1):
            parts.append(f'<col min="{index}" max="{index}" width="{width:.2f}" customWidth="1"/>')
        parts.append("</cols>")
    parts.append("<sheetData>")
    for row_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{row_idx}">')
        for col_idx, value in enumerate(row, start=1):
            header = headers[col_idx - 1] if col_idx <= len(headers) else None
            style = cell_style(header, row_idx in header_rows, value)
            parts.append(cell_xml(row_idx, col_idx, value, style))
        parts.append("</row>")
    parts.append("</sheetData>")
    if autofilter and rows and rows[0] and len(rows) > 1 and 1 in header_rows:
        parts.append(f'<autoFilter ref="A1:{cell_ref(len(rows), len(rows[0]))}"/>')
    if drawing_rid:
        parts.append(f'<drawing r:id="{drawing_rid}"/>')
    parts.append("</worksheet>")
    return "".join(parts)


def num_cache_xml(values: list[float]) -> str:
    parts = [f'<c:ptCount val="{len(values)}"/>']
    for index, value in enumerate(values):
        parts.append(f'<c:pt idx="{index}"><c:v>{float(value):.15g}</c:v></c:pt>')
    return "".join(parts)


def num_ref_xml(ref: str, values: list[float]) -> str:
    return (
        f"<c:numRef><c:f>{xml_escape(ref)}</c:f><c:numCache>"
        "<c:formatCode>General</c:formatCode>"
        f"{num_cache_xml(values)}"
        "</c:numCache></c:numRef>"
    )


def chart_xml(chart: ChartSpec, chart_id: int) -> str:
    x_axis_id = 1000 + chart_id * 2
    y_axis_id = x_axis_id + 1
    y_scaling = '<c:scaling><c:logBase val="10"/><c:orientation val="minMax"/></c:scaling>' if chart.y_log else '<c:scaling><c:orientation val="minMax"/></c:scaling>'
    y_format = "0.00%" if chart.y_percent else "#,##0"
    grayscale_colours = ["1F1F1F", "404040", "595959", "707070", "888888", "A0A0A0", "B8B8B8"]
    grayscale_dashes = ["solid", "dash", "sysDot", "dashDot", "lgDash", "lgDashDot", "lgDashDotDot"]

    def axis_title_xml(value: str | None) -> str:
        if not value:
            return ""
        return (
            "<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p>"
            f"<a:r><a:t>{xml_escape(value)}</a:t></a:r></a:p>"
            "</c:rich></c:tx><c:layout/></c:title>"
        )

    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        "<c:date1904 val=\"0\"/>",
        "<c:chart>",
        "<c:title><c:tx><c:rich><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>",
        xml_escape(chart.title),
        "</a:t></a:r></a:p></c:rich></c:tx><c:layout/></c:title>",
        "<c:plotArea><c:layout/>",
        '<c:scatterChart><c:scatterStyle val="lineMarker"/>',
    ]

    for index, series in enumerate(chart.series):
        line_xml = ""
        marker_style_xml = ""
        if chart.grayscale_styles:
            style_index = series.style_index if series.style_index is not None else index
            colour = grayscale_colours[style_index % len(grayscale_colours)]
            dash = "sysDash" if series.is_fit else grayscale_dashes[style_index % len(grayscale_dashes)]
            line_width = "12700" if series.is_fit else "19050"
            line_xml = (
                f'<c:spPr><a:ln w="{line_width}">'
                f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>'
                f'<a:prstDash val="{dash}"/>'
                "</a:ln></c:spPr>"
            )
            if not series.is_fit:
                marker_style_xml = (
                    "<c:spPr>"
                    f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>'
                    f'<a:ln><a:solidFill><a:srgbClr val="{colour}"/></a:solidFill></a:ln>'
                    "</c:spPr>"
                )
        marker_xml = '<c:marker><c:symbol val="none"/></c:marker>' if series.is_fit else f'<c:marker><c:symbol val="circle"/><c:size val="5"/>{marker_style_xml}</c:marker>'
        parts.extend(
            [
                "<c:ser>",
                f'<c:idx val="{index}"/><c:order val="{index}"/>',
                f"<c:tx><c:v>{xml_escape(series.name)}</c:v></c:tx>",
                line_xml,
                marker_xml,
                "<c:xVal>",
                num_ref_xml(series.x_ref, series.x_values),
                "</c:xVal><c:yVal>",
                num_ref_xml(series.y_ref, series.y_values),
                "</c:yVal>",
                '<c:smooth val="0"/>',
                "</c:ser>",
            ]
        )

    parts.extend(
        [
            f'<c:axId val="{x_axis_id}"/><c:axId val="{y_axis_id}"/>',
            "</c:scatterChart>",
            f'<c:valAx><c:axId val="{x_axis_id}"/>'
            '<c:scaling><c:orientation val="minMax"/></c:scaling>'
            '<c:delete val="0"/><c:axPos val="b"/><c:majorGridlines/>'
            f"{axis_title_xml(chart.x_title)}"
            '<c:numFmt formatCode="0" sourceLinked="0"/>'
            '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
            '<c:tickLblPos val="nextTo"/>'
            f'<c:crossAx val="{y_axis_id}"/><c:crosses val="autoZero"/>'
            "</c:valAx>",
            f'<c:valAx><c:axId val="{y_axis_id}"/>'
            f"{y_scaling}"
            '<c:delete val="0"/><c:axPos val="l"/><c:majorGridlines/>'
            f"{axis_title_xml(chart.y_title)}"
            f'<c:numFmt formatCode="{y_format}" sourceLinked="0"/>'
            '<c:majorTickMark val="out"/><c:minorTickMark val="none"/>'
            '<c:tickLblPos val="nextTo"/>'
            f'<c:crossAx val="{x_axis_id}"/><c:crosses val="autoZero"/>'
            "</c:valAx>",
            "</c:plotArea>",
            '<c:legend><c:legendPos val="r"/><c:layout/></c:legend>',
            '<c:plotVisOnly val="1"/>',
            "</c:chart>",
            "</c:chartSpace>",
        ]
    )
    return "".join(parts)


def drawing_xml(charts: list[tuple[int, ChartSpec]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<xdr:wsDr xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
    ]
    for rel_index, (chart_id, chart) in enumerate(charts, start=1):
        cx = int(chart.width_px * 9525)
        cy = int(chart.height_px * 9525)
        parts.extend(
            [
                "<xdr:oneCellAnchor>",
                "<xdr:from>",
                f"<xdr:col>{chart.anchor_col}</xdr:col><xdr:colOff>0</xdr:colOff>",
                f"<xdr:row>{chart.anchor_row}</xdr:row><xdr:rowOff>0</xdr:rowOff>",
                "</xdr:from>",
                f'<xdr:ext cx="{cx}" cy="{cy}"/>',
                '<xdr:graphicFrame macro="">',
                '<xdr:nvGraphicFramePr><xdr:cNvPr id="',
                str(rel_index),
                '" name="Chart ',
                str(chart_id),
                '"/><xdr:cNvGraphicFramePr/></xdr:nvGraphicFramePr>',
                '<xdr:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/></xdr:xfrm>',
                '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">',
                f'<c:chart r:id="rId{rel_index}"/>',
                "</a:graphicData></a:graphic>",
                "</xdr:graphicFrame>",
                "<xdr:clientData/>",
                "</xdr:oneCellAnchor>",
            ]
        )
    parts.append("</xdr:wsDr>")
    return "".join(parts)


def drawing_rels_xml(charts: list[tuple[int, ChartSpec]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for rel_index, (chart_id, _) in enumerate(charts, start=1):
        parts.append(
            f'<Relationship Id="rId{rel_index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" '
            f'Target="../charts/chart{chart_id}.xml"/>'
        )
    parts.append("</Relationships>")
    return "".join(parts)


def worksheet_rels_xml(drawing_id: int) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" '
        f'Target="../drawings/drawing{drawing_id}.xml"/>'
        "</Relationships>"
    )


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="3">
    <numFmt numFmtId="164" formatCode="0.00%"/>
    <numFmt numFmtId="165" formatCode="#,##0"/>
    <numFmt numFmtId="166" formatCode="0.00"/>
  </numFmts>
  <fonts count="2">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for index, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{xml_escape(name)}" sheetId="{index}" r:id="rId{index}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        + "".join(sheets)
        + "</sheets></workbook>"
    )


def workbook_rels_xml(sheet_count: int) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for index in range(1, sheet_count + 1):
        parts.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    parts.append(
        f'<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    parts.append("</Relationships>")
    return "".join(parts)


def content_types_xml(sheet_count: int, drawing_count: int, chart_count: int) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for index in range(1, sheet_count + 1):
        parts.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    for index in range(1, drawing_count + 1):
        parts.append(
            f'<Override PartName="/xl/drawings/drawing{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.drawing+xml"/>'
        )
    for index in range(1, chart_count + 1):
        parts.append(
            f'<Override PartName="/xl/charts/chart{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>'
        )
    parts.append("</Types>")
    return "".join(parts)


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def doc_props_xml() -> tuple[str, str]:
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>Steganogram probing experiment</dc:title>"
        "<dc:creator>Codex</dc:creator>"
        "</cp:coreProperties>"
    )
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Microsoft Excel</Application>"
        "</Properties>"
    )
    return core, app


def summary_lookup(summary_rows: list[dict]) -> dict[tuple[str, float, float], dict]:
    return {
        (str(row["method"]).upper(), float(row["alpha"]), float(row["probe_budget_percentage"])): row
        for row in summary_rows
    }


def build_used_pixels_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = [
        ["alpha", "HUGO_mean_used_pixels", "HUGO_used_pixel_fraction", "MIPOD_mean_used_pixels", "MIPOD_used_pixel_fraction"],
    ]
    for alpha in ALPHAS:
        hugo = lookup[("HUGO", alpha, BUDGET_PERCENTAGES[0])]
        mipod = lookup[("MIPOD", alpha, BUDGET_PERCENTAGES[0])]
        rows.append(
            [
                alpha,
                float(hugo["mean_used_pixels"]),
                float(hugo["mean_used_pixel_fraction"]),
                float(mipod["mean_used_pixels"]),
                float(mipod["mean_used_pixel_fraction"]),
            ]
        )

    sheet = "UsedPixels"
    x_ref = abs_range(sheet, 2, 1, 1 + len(ALPHAS), 1)
    x_values = [float(alpha) for alpha in ALPHAS]
    charts = [
        ChartSpec(
            sheet_name=sheet,
            title="HUGO: used pixels for stego injection",
            series=[
                ChartSeries("HUGO used pixels", x_ref, abs_range(sheet, 2, 2, 1 + len(ALPHAS), 2), x_values, [float(row[1]) for row in rows[1:]])
            ],
            anchor_row=1,
            anchor_col=7,
            y_percent=False,
        ),
        ChartSpec(
            sheet_name=sheet,
            title="MiPOD: used pixels for stego injection",
            series=[
                ChartSeries("MiPOD used pixels", x_ref, abs_range(sheet, 2, 4, 1 + len(ALPHAS), 4), x_values, [float(row[3]) for row in rows[1:]])
            ],
            anchor_row=21,
            anchor_col=7,
            y_percent=False,
        ),
    ]
    return rows, charts


def build_guessed_carrier_pixels_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "GuessedCarrierPixels"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"{method_label(method)}: mean guessed carrier pixels by probing budget",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    float(lookup[(method, alpha, budget)]["mean_guessed_carrier_pixels"])
                    for alpha in ALPHAS
                ]
            )
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series = [
            ChartSeries(
                alpha_label(alpha),
                x_ref,
                abs_range(sheet, data_start, 2 + alpha_index, data_end, 2 + alpha_index),
                x_values,
                [float(lookup[(method, alpha, budget)]["mean_guessed_carrier_pixels"]) for budget in BUDGET_PERCENTAGES],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: mean guessed carrier pixels",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
                y_percent=False,
            )
        )

    return rows, charts


def build_recall_method_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "CarrierRecall_Methods"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"{method_label(method)}: mean(guessed_carrier_pixels / carrier_pixels) by probing budget",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        header_row = len(rows) + 1
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    float(lookup[(method, alpha, budget)]["mean_carrier_recall"])
                    for alpha in ALPHAS
                ]
            )
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series = [
            ChartSeries(
                alpha_label(alpha),
                x_ref,
                abs_range(sheet, data_start, 2 + alpha_index, data_end, 2 + alpha_index),
                x_values,
                [float(lookup[(method, alpha, budget)]["mean_carrier_recall"]) for budget in BUDGET_PERCENTAGES],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: guessed_carrier_pixels / carrier_pixels, linear y-axis",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
            )
        )
        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: guessed_carrier_pixels / carrier_pixels, logarithmic y-axis",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=22,
                y_log=True,
            )
        )

    return rows, charts


def build_alpha_compare_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "CarrierRecall_AlphaCompare"

    for alpha_index, alpha in enumerate(ALPHAS):
        start_row = 1 + alpha_index * 12
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"Mean(guessed_carrier_pixels / carrier_pixels) comparison for {alpha_label(alpha)}",
                "",
                "",
            ]
        )
        rows.append(
            [
                "probe_budget_percentage",
                "HUGO_mean_guessed_carrier_pixels_div_carrier_pixels",
                "MIPOD_mean_guessed_carrier_pixels_div_carrier_pixels",
            ]
        )
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [
                    budget,
                    float(lookup[("HUGO", alpha, budget)]["mean_carrier_recall"]),
                    float(lookup[("MIPOD", alpha, budget)]["mean_carrier_recall"]),
                ]
            )
        data_end = len(rows)
        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"HUGO vs MiPOD: guessed_carrier_pixels / carrier_pixels, {alpha_label(alpha)}",
                series=[
                    ChartSeries(
                        "HUGO",
                        x_ref,
                        abs_range(sheet, data_start, 2, data_end, 2),
                        x_values,
                        [float(lookup[("HUGO", alpha, budget)]["mean_carrier_recall"]) for budget in BUDGET_PERCENTAGES],
                    ),
                    ChartSeries(
                        "MiPOD",
                        x_ref,
                        abs_range(sheet, data_start, 3, data_end, 3),
                        x_values,
                        [float(lookup[("MIPOD", alpha, budget)]["mean_carrier_recall"]) for budget in BUDGET_PERCENTAGES],
                    ),
                ],
                anchor_row=start_row - 1,
                anchor_col=5,
                width_px=660,
                height_px=300,
            )
        )

    return rows, charts


def build_precision_method_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "PrecisionHitRate_Methods"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"{method_label(method)}: mean(guessed_carrier_pixels / probe_budget_pixels) by probing budget",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    float(lookup[(method, alpha, budget)]["mean_precision_hit_rate"])
                    for alpha in ALPHAS
                ]
            )
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series = [
            ChartSeries(
                alpha_label(alpha),
                x_ref,
                abs_range(sheet, data_start, 2 + alpha_index, data_end, 2 + alpha_index),
                x_values,
                [float(lookup[(method, alpha, budget)]["mean_precision_hit_rate"]) for budget in BUDGET_PERCENTAGES],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: precision hit rate, linear y-axis",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
            )
        )
        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: precision hit rate, logarithmic y-axis",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=22,
                y_log=True,
            )
        )

    return rows, charts


def build_precision_alpha_compare_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "PrecisionHitRate_AlphaCompare"

    for alpha_index, alpha in enumerate(ALPHAS):
        start_row = 1 + alpha_index * 12
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"Mean(guessed_carrier_pixels / probe_budget_pixels) comparison for {alpha_label(alpha)}",
                "",
                "",
            ]
        )
        rows.append(
            [
                "probe_budget_percentage",
                "HUGO_mean_precision_hit_rate",
                "MIPOD_mean_precision_hit_rate",
            ]
        )
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [
                    budget,
                    float(lookup[("HUGO", alpha, budget)]["mean_precision_hit_rate"]),
                    float(lookup[("MIPOD", alpha, budget)]["mean_precision_hit_rate"]),
                ]
            )
        data_end = len(rows)
        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"HUGO vs MiPOD: precision hit rate, {alpha_label(alpha)}",
                series=[
                    ChartSeries(
                        "HUGO",
                        x_ref,
                        abs_range(sheet, data_start, 2, data_end, 2),
                        x_values,
                        [float(lookup[("HUGO", alpha, budget)]["mean_precision_hit_rate"]) for budget in BUDGET_PERCENTAGES],
                    ),
                    ChartSeries(
                        "MiPOD",
                        x_ref,
                        abs_range(sheet, data_start, 3, data_end, 3),
                        x_values,
                        [float(lookup[("MIPOD", alpha, budget)]["mean_precision_hit_rate"]) for budget in BUDGET_PERCENTAGES],
                    ),
                ],
                anchor_row=start_row - 1,
                anchor_col=5,
                width_px=660,
                height_px=300,
            )
        )

    return rows, charts


def build_guessed_per_probed_sheet(summary_rows: list[dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    lookup = summary_lookup(summary_rows)
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "GuessedPerProbedPixels"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append(
            [
                f"{method_label(method)}: mean guessed_carrier_pixels / probed_pixels by probing budget",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    float(lookup[(method, alpha, budget)]["mean_precision_hit_rate"])
                    for alpha in ALPHAS
                ]
            )
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series = [
            ChartSeries(
                alpha_label(alpha),
                x_ref,
                abs_range(sheet, data_start, 2 + alpha_index, data_end, 2 + alpha_index),
                x_values,
                [float(lookup[(method, alpha, budget)]["mean_precision_hit_rate"]) for budget in BUDGET_PERCENTAGES],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: guessed_carrier_pixels / probed_pixels",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
            )
        )

    return rows, charts


def load_metadata() -> dict:
    if not METADATA_JSON.exists():
        return {}
    return json.loads(METADATA_JSON.read_text(encoding="utf-8"))


def build_notes_sheet(raw_count: int, summary_count: int, metadata: dict) -> list[list[object]]:
    return [
        ["Item", "Value"],
        ["Purpose", "Probing-only experiment for simulated HUGO/MiPOD steganograms."],
        ["Raw rows", raw_count],
        ["Summary rows", summary_count],
        ["Probe budgets", ", ".join(f"{int(p)}%" for p in BUDGET_PERCENTAGES)],
        ["Probe distribution", experiment.PROBE_DISTRIBUTION],
        ["Configured probe seed", metadata.get("configured_probe_seed", "")],
        ["Run probe seed", metadata.get("run_probe_seed", "")],
        ["Probe seed mode", metadata.get("probe_seed_mode", "")],
        ["Probe image source", experiment.PROBE_IMAGE_SOURCE],
        ["Uniqueness rule", "Each image/config uses one non-replacement probe order up to 50%; smaller budgets are prefixes of that same unique order."],
        ["Probe budget", "Percentage of total input-image pixels, e.g. 1% of a 512x512 image is 2621 pixels."],
        ["Guessed carrier pixels", "Count of probed pixels that were carrier/changing pixels. Graphs use mean_guessed_carrier_pixels."],
        ["Carrier recall", "guessed_carrier_pixels / carrier_pixels"],
        ["Carrier recall aggregation", "Charts use mean_carrier_recall, i.e. the mean per-image value of guessed_carrier_pixels / carrier_pixels."],
        ["Precision hit rate", "guessed_carrier_pixels / probe_budget_pixels"],
        ["Precision hit rate aggregation", "Charts use mean_precision_hit_rate, i.e. the mean per-image value of guessed_carrier_pixels / probe_budget_pixels."],
        ["Guessed/probed pixels", "Same value as precision_hit_rate: guessed_carrier_pixels / probe_budget_pixels."],
        ["Incremental precision", "new guessed carrier pixels / new pixels added since the previous budget"],
    ]


def build_workbook_data() -> tuple[dict[str, list[list[object]]], dict[str, list[ChartSpec]]]:
    if not RAW_CSV.exists() or not SUMMARY_CSV.exists():
        experiment.main()

    raw_headers, raw_rows = read_csv_rows(RAW_CSV)
    summary_headers, summary_rows = read_csv_rows(SUMMARY_CSV)
    metadata = load_metadata()

    used_rows, used_charts = build_used_pixels_sheet(summary_rows)
    guessed_rows, guessed_charts = build_guessed_carrier_pixels_sheet(summary_rows)
    method_rows, method_charts = build_recall_method_sheet(summary_rows)
    alpha_rows, alpha_charts = build_alpha_compare_sheet(summary_rows)
    precision_method_rows, precision_method_charts = build_precision_method_sheet(summary_rows)
    precision_alpha_rows, precision_alpha_charts = build_precision_alpha_compare_sheet(summary_rows)
    guessed_per_probed_rows, guessed_per_probed_charts = build_guessed_per_probed_sheet(summary_rows)

    sheets = {
        "Notes": build_notes_sheet(len(raw_rows), len(summary_rows), metadata),
        "RawData": rows_from_dicts(raw_headers, raw_rows),
        "Summary": rows_from_dicts(summary_headers, summary_rows),
        "UsedPixels": used_rows,
        "GuessedCarrierPixels": guessed_rows,
        "CarrierRecall_Methods": method_rows,
        "CarrierRecall_AlphaCompare": alpha_rows,
        "PrecisionHitRate_Methods": precision_method_rows,
        "PrecisionHitRate_AlphaCompare": precision_alpha_rows,
        "GuessedPerProbedPixels": guessed_per_probed_rows,
    }
    charts = {
        "UsedPixels": used_charts,
        "GuessedCarrierPixels": guessed_charts,
        "CarrierRecall_Methods": method_charts,
        "CarrierRecall_AlphaCompare": alpha_charts,
        "PrecisionHitRate_Methods": precision_method_charts,
        "PrecisionHitRate_AlphaCompare": precision_alpha_charts,
        "GuessedPerProbedPixels": guessed_per_probed_charts,
    }
    return sheets, charts


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]], charts_by_sheet: dict[str, list[ChartSpec]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())
    drawing_by_sheet: dict[str, int] = {}
    chart_id_by_spec: list[tuple[int, ChartSpec]] = []

    drawing_id = 0
    chart_id = 0
    for sheet_name in sheet_names:
        sheet_charts = charts_by_sheet.get(sheet_name, [])
        if sheet_charts:
            drawing_id += 1
            drawing_by_sheet[sheet_name] = drawing_id
            for chart in sheet_charts:
                chart_id += 1
                chart_id_by_spec.append((chart_id, chart))

    charts_grouped_by_drawing: dict[int, list[tuple[int, ChartSpec]]] = {}
    for chart_id_value, chart in chart_id_by_spec:
        charts_grouped_by_drawing.setdefault(drawing_by_sheet[chart.sheet_name], []).append((chart_id_value, chart))

    core_props, app_props = doc_props_xml()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml(len(sheet_names), drawing_id, chart_id))
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("docProps/core.xml", core_props)
        archive.writestr("docProps/app.xml", app_props)
        archive.writestr("xl/workbook.xml", workbook_xml(sheet_names))
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheet_names)))
        archive.writestr("xl/styles.xml", styles_xml())

        for sheet_index, sheet_name in enumerate(sheet_names, start=1):
            drawing = drawing_by_sheet.get(sheet_name)
            widths = [18.0] * max((len(row) for row in sheets[sheet_name] if row), default=1)
            if sheet_name == "RawData":
                widths = [22.0] * len(sheets[sheet_name][0])
            archive.writestr(
                f"xl/worksheets/sheet{sheet_index}.xml",
                worksheet_xml(
                    sheets[sheet_name],
                    drawing_rid="rId1" if drawing else None,
                    widths=widths,
                    freeze_top_row=sheet_name in {"RawData", "Summary", "Notes"},
                    autofilter=sheet_name in {"RawData", "Summary", "Notes"},
                ),
            )
            if drawing:
                archive.writestr(f"xl/worksheets/_rels/sheet{sheet_index}.xml.rels", worksheet_rels_xml(drawing))

        for current_drawing_id, drawing_charts in charts_grouped_by_drawing.items():
            archive.writestr(f"xl/drawings/drawing{current_drawing_id}.xml", drawing_xml(drawing_charts))
            archive.writestr(f"xl/drawings/_rels/drawing{current_drawing_id}.xml.rels", drawing_rels_xml(drawing_charts))

        for chart_id_value, chart in chart_id_by_spec:
            archive.writestr(f"xl/charts/chart{chart_id_value}.xml", chart_xml(chart, chart_id_value))


def main() -> None:
    sheets, charts = build_workbook_data()
    write_xlsx(OUTPUT_XLSX, sheets, charts)
    print(f"Wrote workbook to {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from generate_probing_excel import ChartSeries, ChartSpec, abs_range, rows_from_dicts, write_xlsx


SUMMARY_CSV = Path("experiments/probing_only/probe_summary.csv")
METADATA_JSON = Path("experiments/probing_only/metadata.json")
OUTPUT_XLSX = Path("outputs/019dd41c-14e1-7142-8f87-d2b78c44a471/texture_oriented_figures_for_theses.xlsx")
METHODS = ["HUGO", "MIPOD"]
ALPHAS = [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
BUDGETS = [1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def lookup(rows: list[dict[str, str]]) -> dict[tuple[str, float, float], dict[str, str]]:
    return {
        (row["method"].upper(), float(row["alpha"]), float(row["probe_budget_percentage"])): row
        for row in rows
    }


def degree_label(alpha: float) -> str:
    return f"{int(round(100 * alpha))}%"


def fit_power_model(
    values: dict[tuple[str, float, float], dict[str, str]],
    method: str,
    alpha: float,
) -> tuple[float, float, float]:
    x_values = [math.log(budget) for budget in BUDGETS]
    y_values = [
        math.log(float(values[(method, alpha, budget)]["mean_guessed_carrier_pixels"]))
        for budget in BUDGETS
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


def graph_sheet(
    values: dict[tuple[str, float, float], dict[str, str]], method: str, sheet: str
) -> tuple[list[list[object]], list[ChartSpec]]:
    fit_params = {
        alpha: fit_power_model(values, method, alpha)
        for alpha in ALPHAS
    }
    rows: list[list[object]] = [
        [f"{method}: correctly identified carrier pixels H(B), texture-oriented probing"],
        ["Probing budget (%)"] + [
            label
            for alpha in ALPHAS
            for label in (degree_label(alpha), f"{degree_label(alpha)} fit")
        ],
    ]
    for budget in BUDGETS:
        rows.append(
            [budget]
            + [
                value
                for alpha in ALPHAS
                for value in (
                    float(values[(method, alpha, budget)]["mean_guessed_carrier_pixels"]),
                    fit_params[alpha][0] * (budget ** fit_params[alpha][1]),
                )
            ]
        )
    rows.append([])
    rows.append(["Power model", "H(B)=aB^p"])
    rows.append(["Embedding level", "a", "p", "R²"])
    for alpha in ALPHAS:
        coefficient, exponent, r_squared = fit_params[alpha]
        rows.append([degree_label(alpha), coefficient, exponent, r_squared])

    x_ref = abs_range(sheet, 3, 1, 2 + len(BUDGETS), 1)
    x_values = BUDGETS
    series = []
    for alpha_index, alpha in enumerate(ALPHAS):
        observed_col = 2 + 2 * alpha_index
        fitted_col = observed_col + 1
        coefficient, exponent, r_squared = fit_params[alpha]
        series.append(
            ChartSeries(
                degree_label(alpha),
                x_ref,
                abs_range(sheet, 3, observed_col, 2 + len(BUDGETS), observed_col),
                x_values,
                [float(values[(method, alpha, budget)]["mean_guessed_carrier_pixels"]) for budget in BUDGETS],
                style_index=alpha_index,
            )
        )
        series.append(
            ChartSeries(
                f"{degree_label(alpha)} fit",
                x_ref,
                abs_range(sheet, 3, fitted_col, 2 + len(BUDGETS), fitted_col),
                x_values,
                [coefficient * (budget ** exponent) for budget in BUDGETS],
                style_index=alpha_index,
                is_fit=True,
            )
        )
    chart = ChartSpec(
        sheet_name=sheet,
        title=f"{method}: correctly identified carrier pixels and power fit",
        series=series,
        anchor_row=1,
        anchor_col=17,
        width_px=980,
        height_px=520,
        y_log=True,
        y_percent=False,
        x_title="Probing budget (% of image pixels)",
        y_title="Correctly identified carrier pixels H(B), log scale",
        grayscale_styles=True,
    )
    return rows, [chart]


def notes_sheet(metadata: dict, summary_rows: int) -> list[list[object]]:
    return [
        ["Item", "Value"],
        ["Purpose", "Editable source data and native Excel charts used as Figures 1 and 2 in the theses."],
        ["Proposed distribution", "Texture-oriented probing (the source metadata identifier is texture_energy)."],
        ["Displayed outcome", "H(B): mean number of correctly identified carrier pixels among 100 images."],
        ["Approximation", "Each chart includes fitted curves for H(B)=aB^p and a parameter table with a, p, and R²."],
        ["X-axis", "Probing budget B as a percentage of all 512 x 512 image pixels."],
        ["Y-axis", "Logarithmic axis for H(B); the charts are editable Excel chart objects."],
        ["Series", "Embedding level δ_α: 3%, 5%, 10%, 20%, 30%, 40%, and 50%."],
        ["Display of pixel values", "Cells retain mean values; an integer pixel display can be applied when formatting figures."],
        ["Summary rows", summary_rows],
        ["Embedding methods", "HUGO and MiPOD single-channel simulation using conseal."],
        ["Embedding seed", metadata.get("embed_seed", "")],
        ["Probing seed", metadata.get("probe_seed", "")],
        ["Sampling rule", "Weighted sampling without replacement; smaller budgets are prefixes of one order."],
        ["Source CSV", str(SUMMARY_CSV.resolve())],
    ]


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary data not found: {SUMMARY_CSV.resolve()}")

    headers, summary_rows = read_csv(SUMMARY_CSV)
    metadata = json.loads(METADATA_JSON.read_text(encoding="utf-8")) if METADATA_JSON.exists() else {}
    values = lookup(summary_rows)

    hugo_rows, hugo_charts = graph_sheet(values, "HUGO", "Figure1_HUGO")
    mipod_rows, mipod_charts = graph_sheet(values, "MIPOD", "Figure2_MiPOD")
    sheets = {
        "Figure1_HUGO": hugo_rows,
        "Figure2_MiPOD": mipod_rows,
        "SourceSummary": rows_from_dicts(headers, summary_rows),
        "Notes": notes_sheet(metadata, len(summary_rows)),
    }
    charts = {
        "Figure1_HUGO": hugo_charts,
        "Figure2_MiPOD": mipod_charts,
    }
    write_xlsx(OUTPUT_XLSX, sheets, charts)
    print(f"Wrote workbook to {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()

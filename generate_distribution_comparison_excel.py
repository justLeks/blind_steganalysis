from __future__ import annotations

import csv
import json
from pathlib import Path

from generate_probing_excel import (
    ALPHAS,
    BUDGET_PERCENTAGES,
    ChartSeries,
    ChartSpec,
    abs_range,
    alpha_label,
    method_label,
    write_xlsx,
)


# Edit these settings and run: python3 generate_distribution_comparison_excel.py
UNIFORM_ROOT = Path("experiments/probing_uniform")
TEXTURE_ROOT = Path("experiments/probing_only")
OUTPUT_ROOT = Path("experiments/probe_distribution_comparison")
OUTPUT_XLSX = OUTPUT_ROOT / "probe_distribution_comparison.xlsx"

METHODS = ["HUGO", "MIPOD"]
DISTRIBUTIONS = {
    "uniform": UNIFORM_ROOT / "probe_summary.csv",
    "texture_energy": TEXTURE_ROOT / "probe_summary.csv",
}


def read_summary(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_lookup(rows_by_distribution: dict[str, list[dict]]) -> dict[tuple[str, str, float, float], dict]:
    lookup: dict[tuple[str, str, float, float], dict] = {}
    for distribution, rows in rows_by_distribution.items():
        for row in rows:
            key = (
                distribution,
                str(row["method"]).upper(),
                float(row["alpha"]),
                float(row["probe_budget_percentage"]),
            )
            lookup[key] = row
    return lookup


def get_metric(
    lookup: dict[tuple[str, str, float, float], dict],
    distribution: str,
    method: str,
    alpha: float,
    budget: float,
    metric: str,
) -> float:
    return float(lookup[(distribution, method.upper(), alpha, budget)][metric])


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_notes_sheet(rows_by_distribution: dict[str, list[dict]]) -> list[list[object]]:
    metadata_rows = [["Distribution", "Metadata path", "Run probe seed", "Seed mode"]]
    for distribution, root in [("uniform", UNIFORM_ROOT), ("texture_energy", TEXTURE_ROOT)]:
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        metadata_rows.append(
            [
                distribution,
                str(metadata_path.resolve()),
                metadata.get("run_probe_seed", metadata.get("probe_seed", "")),
                metadata.get("probe_seed_mode", ""),
            ]
        )

    rows = [
        ["Item", "Value"],
        ["Purpose", "Compare uniform random probing against texture-energy probing."],
        ["Uniform source", str(UNIFORM_ROOT.resolve())],
        ["Texture-energy source", str(TEXTURE_ROOT.resolve())],
        ["Output workbook", str(OUTPUT_XLSX.resolve())],
        ["Comparison unit", "Mean values grouped by method, alpha, and probing budget."],
        ["Guessed carrier pixels", "mean_guessed_carrier_pixels"],
        ["Recall lift", "texture_energy mean_carrier_recall / uniform mean_carrier_recall"],
        ["Precision hit rate", "mean_precision_hit_rate = guessed_carrier_pixels / probe_budget_pixels"],
        ["Delta guessed pixels", "texture_energy mean_guessed_carrier_pixels - uniform mean_guessed_carrier_pixels"],
        ["Probe budgets", ", ".join(f"{int(value)}%" for value in BUDGET_PERCENTAGES)],
        ["Alpha values", ", ".join(str(alpha) for alpha in ALPHAS)],
        [],
    ]
    rows.extend(metadata_rows)
    rows.append([])
    rows.append(["Distribution", "Summary rows"])
    for distribution, rows_for_distribution in rows_by_distribution.items():
        rows.append([distribution, len(rows_for_distribution)])
    return rows


def build_comparison_data_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> list[list[object]]:
    rows: list[list[object]] = [
        [
            "method",
            "alpha",
            "probe_budget_percentage",
            "uniform_mean_carrier_pixels",
            "texture_mean_carrier_pixels",
            "uniform_mean_guessed_carrier_pixels",
            "texture_mean_guessed_carrier_pixels",
            "delta_guessed_texture_minus_uniform",
            "uniform_mean_carrier_recall",
            "texture_mean_carrier_recall",
            "recall_lift_texture_over_uniform",
            "uniform_mean_precision_hit_rate",
            "texture_mean_precision_hit_rate",
            "precision_lift_texture_over_uniform",
        ]
    ]
    for method in METHODS:
        for alpha in ALPHAS:
            for budget in BUDGET_PERCENTAGES:
                uniform_carriers = get_metric(lookup, "uniform", method, alpha, budget, "mean_carrier_pixels")
                texture_carriers = get_metric(lookup, "texture_energy", method, alpha, budget, "mean_carrier_pixels")
                uniform_guessed = get_metric(lookup, "uniform", method, alpha, budget, "mean_guessed_carrier_pixels")
                texture_guessed = get_metric(lookup, "texture_energy", method, alpha, budget, "mean_guessed_carrier_pixels")
                uniform_recall = get_metric(lookup, "uniform", method, alpha, budget, "mean_carrier_recall")
                texture_recall = get_metric(lookup, "texture_energy", method, alpha, budget, "mean_carrier_recall")
                uniform_precision = get_metric(lookup, "uniform", method, alpha, budget, "mean_precision_hit_rate")
                texture_precision = get_metric(lookup, "texture_energy", method, alpha, budget, "mean_precision_hit_rate")
                rows.append(
                    [
                        method_label(method),
                        alpha,
                        budget,
                        uniform_carriers,
                        texture_carriers,
                        uniform_guessed,
                        texture_guessed,
                        texture_guessed - uniform_guessed,
                        uniform_recall,
                        texture_recall,
                        safe_ratio(texture_recall, uniform_recall),
                        uniform_precision,
                        texture_precision,
                        safe_ratio(texture_precision, uniform_precision),
                    ]
                )
    return rows


def build_guessed_overview_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "GuessedPixels_Overview"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 26
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append([f"{method_label(method)}: mean guessed carrier pixels, uniform vs texture-energy", ""])
        header = ["probe_budget_percentage"]
        for distribution in DISTRIBUTIONS:
            header.extend([f"{distribution}_{alpha_label(alpha)}" for alpha in ALPHAS])
        rows.append(header)
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            row = [budget]
            for distribution in DISTRIBUTIONS:
                row.extend(
                    [
                        get_metric(lookup, distribution, method, alpha, budget, "mean_guessed_carrier_pixels")
                        for alpha in ALPHAS
                    ]
                )
            rows.append(row)
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series: list[ChartSeries] = []
        col = 2
        for distribution in DISTRIBUTIONS:
            for alpha in ALPHAS:
                series.append(
                    ChartSeries(
                        f"{distribution} {alpha_label(alpha)}",
                        x_ref,
                        abs_range(sheet, data_start, col, data_end, col),
                        x_values,
                        [
                            get_metric(lookup, distribution, method, alpha, budget, "mean_guessed_carrier_pixels")
                            for budget in BUDGET_PERCENTAGES
                        ],
                    )
                )
                col += 1

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: absolute guessed carrier pixels",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=18,
                width_px=820,
                height_px=420,
                y_percent=False,
            )
        )
    return rows, charts


def build_guessed_by_alpha_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "GuessedPixels_ByAlpha"

    for alpha_index, alpha in enumerate(ALPHAS):
        start_row = 1 + alpha_index * 13
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append([f"Mean guessed carrier pixels for {alpha_label(alpha)}", "", "", "", ""])
        rows.append(
            [
                "probe_budget_percentage",
                "HUGO_uniform",
                "HUGO_texture_energy",
                "MiPOD_uniform",
                "MiPOD_texture_energy",
            ]
        )
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [
                    budget,
                    get_metric(lookup, "uniform", "HUGO", alpha, budget, "mean_guessed_carrier_pixels"),
                    get_metric(lookup, "texture_energy", "HUGO", alpha, budget, "mean_guessed_carrier_pixels"),
                    get_metric(lookup, "uniform", "MIPOD", alpha, budget, "mean_guessed_carrier_pixels"),
                    get_metric(lookup, "texture_energy", "MIPOD", alpha, budget, "mean_guessed_carrier_pixels"),
                ]
            )
        data_end = len(rows)
        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"Guessed carrier pixels: {alpha_label(alpha)}",
                series=[
                    ChartSeries("HUGO uniform", x_ref, abs_range(sheet, data_start, 2, data_end, 2), x_values, [float(row[1]) for row in rows[data_start - 1 : data_end]]),
                    ChartSeries("HUGO texture_energy", x_ref, abs_range(sheet, data_start, 3, data_end, 3), x_values, [float(row[2]) for row in rows[data_start - 1 : data_end]]),
                    ChartSeries("MiPOD uniform", x_ref, abs_range(sheet, data_start, 4, data_end, 4), x_values, [float(row[3]) for row in rows[data_start - 1 : data_end]]),
                    ChartSeries("MiPOD texture_energy", x_ref, abs_range(sheet, data_start, 5, data_end, 5), x_values, [float(row[4]) for row in rows[data_start - 1 : data_end]]),
                ],
                anchor_row=start_row - 1,
                anchor_col=7,
                width_px=720,
                height_px=300,
                y_percent=False,
            )
        )
    return rows, charts


def build_recall_lift_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "RecallLift"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append([f"{method_label(method)}: texture-energy recall lift over uniform", "", "", "", "", "", "", ""])
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    safe_ratio(
                        get_metric(lookup, "texture_energy", method, alpha, budget, "mean_carrier_recall"),
                        get_metric(lookup, "uniform", method, alpha, budget, "mean_carrier_recall"),
                    )
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
                [
                    safe_ratio(
                        get_metric(lookup, "texture_energy", method, alpha, budget, "mean_carrier_recall"),
                        get_metric(lookup, "uniform", method, alpha, budget, "mean_carrier_recall"),
                    )
                    for budget in BUDGET_PERCENTAGES
                ],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: recall lift, texture_energy / uniform",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
                y_percent=False,
            )
        )
    return rows, charts


def build_precision_compare_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "PrecisionCompare"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 26
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append([f"{method_label(method)}: guessed_carrier_pixels / probed_pixels", ""])
        header = ["probe_budget_percentage"]
        for distribution in DISTRIBUTIONS:
            header.extend([f"{distribution}_{alpha_label(alpha)}" for alpha in ALPHAS])
        rows.append(header)
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            row = [budget]
            for distribution in DISTRIBUTIONS:
                row.extend(
                    [
                        get_metric(lookup, distribution, method, alpha, budget, "mean_precision_hit_rate")
                        for alpha in ALPHAS
                    ]
                )
            rows.append(row)
        data_end = len(rows)

        x_ref = abs_range(sheet, data_start, 1, data_end, 1)
        x_values = [float(budget) for budget in BUDGET_PERCENTAGES]
        series: list[ChartSeries] = []
        col = 2
        for distribution in DISTRIBUTIONS:
            for alpha in ALPHAS:
                series.append(
                    ChartSeries(
                        f"{distribution} {alpha_label(alpha)}",
                        x_ref,
                        abs_range(sheet, data_start, col, data_end, col),
                        x_values,
                        [
                            get_metric(lookup, distribution, method, alpha, budget, "mean_precision_hit_rate")
                            for budget in BUDGET_PERCENTAGES
                        ],
                    )
                )
                col += 1

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: guessed carrier pixels / probed pixels",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=18,
                width_px=820,
                height_px=420,
            )
        )
    return rows, charts


def build_delta_guessed_sheet(lookup: dict[tuple[str, str, float, float], dict]) -> tuple[list[list[object]], list[ChartSpec]]:
    rows: list[list[object]] = []
    charts: list[ChartSpec] = []
    sheet = "DeltaGuessed"

    for block_index, method in enumerate(METHODS):
        start_row = 1 + block_index * 24
        while len(rows) < start_row - 1:
            rows.append([])
        rows.append([f"{method_label(method)}: extra guessed pixels, texture-energy minus uniform", "", "", "", "", "", "", ""])
        rows.append(["probe_budget_percentage"] + [alpha_label(alpha) for alpha in ALPHAS])
        data_start = len(rows) + 1
        for budget in BUDGET_PERCENTAGES:
            rows.append(
                [budget]
                + [
                    get_metric(lookup, "texture_energy", method, alpha, budget, "mean_guessed_carrier_pixels")
                    - get_metric(lookup, "uniform", method, alpha, budget, "mean_guessed_carrier_pixels")
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
                [
                    get_metric(lookup, "texture_energy", method, alpha, budget, "mean_guessed_carrier_pixels")
                    - get_metric(lookup, "uniform", method, alpha, budget, "mean_guessed_carrier_pixels")
                    for budget in BUDGET_PERCENTAGES
                ],
            )
            for alpha_index, alpha in enumerate(ALPHAS)
        ]

        charts.append(
            ChartSpec(
                sheet_name=sheet,
                title=f"{method_label(method)}: extra guessed carrier pixels",
                series=series,
                anchor_row=start_row - 1,
                anchor_col=10,
                y_percent=False,
            )
        )
    return rows, charts


def build_workbook_data() -> tuple[dict[str, list[list[object]]], dict[str, list[ChartSpec]]]:
    rows_by_distribution = {
        distribution: read_summary(path)
        for distribution, path in DISTRIBUTIONS.items()
    }
    lookup = build_lookup(rows_by_distribution)

    guessed_overview_rows, guessed_overview_charts = build_guessed_overview_sheet(lookup)
    guessed_alpha_rows, guessed_alpha_charts = build_guessed_by_alpha_sheet(lookup)
    recall_lift_rows, recall_lift_charts = build_recall_lift_sheet(lookup)
    precision_rows, precision_charts = build_precision_compare_sheet(lookup)
    delta_rows, delta_charts = build_delta_guessed_sheet(lookup)

    sheets = {
        "Notes": build_notes_sheet(rows_by_distribution),
        "ComparisonData": build_comparison_data_sheet(lookup),
        "GuessedPixels_Overview": guessed_overview_rows,
        "GuessedPixels_ByAlpha": guessed_alpha_rows,
        "RecallLift": recall_lift_rows,
        "PrecisionCompare": precision_rows,
        "DeltaGuessed": delta_rows,
    }
    charts = {
        "GuessedPixels_Overview": guessed_overview_charts,
        "GuessedPixels_ByAlpha": guessed_alpha_charts,
        "RecallLift": recall_lift_charts,
        "PrecisionCompare": precision_charts,
        "DeltaGuessed": delta_charts,
    }
    return sheets, charts


def main() -> None:
    missing = [str(path) for path in DISTRIBUTIONS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing distribution summary CSV files: {missing}")

    sheets, charts = build_workbook_data()
    write_xlsx(OUTPUT_XLSX, sheets, charts)
    print(f"Wrote workbook to {OUTPUT_XLSX.resolve()}")


if __name__ == "__main__":
    main()

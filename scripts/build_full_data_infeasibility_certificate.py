#!/usr/bin/env python3
"""Construct a Farkas certificate proving full-data hard-margin infeasibility."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from linear_hard_margin_svm import FEATURES, Preprocessor, signed_design  # noqa: E402


DATA_PATH = ROOT / "data" / "cgh_pa_dataset.csv"
CSV_PATH = ROOT / "output" / "csv" / "hard_margin_infeasibility_certificate.csv"
SOLVER_CSV_PATH = ROOT / "output" / "csv" / "full_data_solver_audit.csv"
HULL_CSV_PATH = ROOT / "output" / "csv" / "full_data_convex_hull_point.csv"
AUDIT_CSV_PATH = ROOT / "output" / "csv" / "full_data_infeasibility_audit.csv"
TABLE_DIR = ROOT / "output" / "tables"
SUMMARY_PATH = ROOT / "output" / "figures" / "full_data_infeasibility_certificate.txt"
FIGURE_STEM = ROOT / "output" / "figures" / "full_data_infeasibility_certificate"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def compute_certificate() -> dict[str, object]:
    """Find lambda >= 0 with U.T @ lambda = 0 and sum(lambda) = 1."""
    frame = pd.read_csv(DATA_PATH)
    raw_features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    _, standardized = Preprocessor.fit(raw_features)
    labels = np.where(frame["Model 1 Target"].eq("UPA"), 1.0, -1.0)
    design = signed_design(standardized, labels)

    feasibility_result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-design,
        b_ub=-np.ones(len(design)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    if feasibility_result.success:
        raise RuntimeError("Expected the full-data hard-margin system to be infeasible.")

    equality_matrix = np.vstack([design.T, np.ones(len(design))])
    equality_target = np.r_[np.zeros(design.shape[1]), 1.0]
    result = linprog(
        np.zeros(len(design)),
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=[(0.0, None)] * len(design),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Farkas-certificate LP failed: {result.message}")

    weights = result.x
    active = np.flatnonzero(weights > 1e-9)
    positive = labels == 1.0
    negative = labels == -1.0
    positive_mass = float(weights[positive].sum())
    negative_mass = float(weights[negative].sum())

    positive_coefficients = weights[positive] / positive_mass
    negative_coefficients = weights[negative] / negative_mass
    positive_combination = positive_coefficients @ standardized[positive]
    negative_combination = negative_coefficients @ standardized[negative]

    certificate_residual = float(np.max(np.abs(design.T @ weights)))
    hull_residual = float(
        np.max(np.abs(positive_combination - negative_combination))
    )

    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(positive_mass, 0.5, atol=1e-12)
    np.testing.assert_allclose(negative_mass, 0.5, atol=1e-12)
    np.testing.assert_allclose(design.T @ weights, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        positive_combination, negative_combination, atol=1e-12
    )

    certificate = pd.DataFrame(
        {
            "observation_number": active + 1,
            "PatientID": frame.iloc[active]["PatientID"].to_numpy(),
            "class": frame.iloc[active]["Model 1 Target"].to_numpy(),
            "label_y": labels[active].astype(int),
            "lambda_full_precision": weights[active],
            "lambda_display_2dp": [f"{value:.2f}" for value in weights[active]],
        }
    )

    return {
        "certificate": certificate,
        "positive_combination": positive_combination,
        "negative_combination": negative_combination,
        "certificate_residual": certificate_residual,
        "hull_residual": hull_residual,
        "positive_mass": positive_mass,
        "negative_mass": negative_mass,
        "solver_message": str(feasibility_result.message),
        "solver_status": int(feasibility_result.status),
    }


def _write_latex_tables(certificate: pd.DataFrame) -> None:
    """Write compact LaTeX fragments from the same data saved to CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    solver_table = r"""\begin{tabular}{ll}
\toprule
Item & Solver result \\
\midrule
Decision variables $(w_1,\ldots,w_{10},b)$ & 11 \\
Hard-margin constraints & 114 \\
Solver & SciPy HiGHS \\
HiGHS model status & Status 8: Infeasible \\
Primal parameters $(w,b)$ returned & None \\
Primal objective $\frac12\lVert w\rVert_2^2$ & Not defined \\
\bottomrule
\end{tabular}
"""
    (TABLE_DIR / "full_data_solver_audit_table.tex").write_text(
        solver_table, encoding="utf-8"
    )

    rows = []
    for left_index in range(6):
        left = certificate.iloc[left_index]
        right = certificate.iloc[left_index + 6]
        rows.append(
            f"{left['PatientID']} & {left['class']} & {left['lambda_display_2dp']} & "
            f"{right['PatientID']} & {right['class']} & {right['lambda_display_2dp']} \\\\"
        )
    farkas_table = "\n".join(
        [
            r"\begin{tabular}{llr@{\qquad}llr}",
            r"\toprule",
            r"Patient & Class & $\lambda_i$ & Patient & Class & $\lambda_i$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (TABLE_DIR / "farkas_certificate_table.tex").write_text(
        farkas_table, encoding="utf-8"
    )


def save_outputs(result: dict[str, object]) -> None:
    """Save the sparse certificate, numerical audit, and explanatory figure."""
    certificate = result["certificate"]
    if not isinstance(certificate, pd.DataFrame):
        raise TypeError("certificate must be a DataFrame")
    positive_combination = np.asarray(result["positive_combination"], dtype=float)
    negative_combination = np.asarray(result["negative_combination"], dtype=float)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    certificate.to_csv(CSV_PATH, index=False, float_format="%.15g")

    solver_audit = pd.DataFrame(
        [
            {
                "n_observations": 114,
                "n_features": len(FEATURES),
                "n_decision_variables": len(FEATURES) + 1,
                "n_margin_constraints": 114,
                "solver": "SciPy HiGHS",
                "solver_status_code": int(result["solver_status"]),
                "model_status": "Infeasible",
                "primal_parameters_returned": "No",
                "primal_objective": "Not defined",
                "solver_message": str(result["solver_message"]),
            }
        ]
    )
    solver_audit.to_csv(SOLVER_CSV_PATH, index=False)

    hull_point = pd.DataFrame(
        {
            "feature": FEATURES,
            "upa_convex_combination_full_precision": positive_combination,
            "bpa_convex_combination_full_precision": negative_combination,
            "absolute_difference_full_precision": np.abs(
                positive_combination - negative_combination
            ),
            "upa_display_2dp": [f"{value:.2f}" for value in positive_combination],
            "bpa_display_2dp": [f"{value:.2f}" for value in negative_combination],
        }
    )
    hull_point.to_csv(HULL_CSV_PATH, index=False, float_format="%.15g")

    audit = pd.DataFrame(
        [
            ("sum_lambda", float(certificate["lambda_full_precision"].sum()), "1.00"),
            ("upa_lambda_mass", float(result["positive_mass"]), "0.50"),
            ("bpa_lambda_mass", float(result["negative_mass"]), "0.50"),
            (
                "max_abs_U_transpose_lambda",
                float(result["certificate_residual"]),
                f"{float(result['certificate_residual']):.2e}",
            ),
            (
                "max_convex_hull_coordinate_difference",
                float(result["hull_residual"]),
                f"{float(result['hull_residual']):.2e}",
            ),
        ],
        columns=["metric", "full_precision_value", "display_value"],
    )
    audit.to_csv(AUDIT_CSV_PATH, index=False, float_format="%.15g")
    _write_latex_tables(certificate)

    summary = [
        "Full-data linear hard-margin infeasibility certificate",
        "========================================================",
        f"Samples: 114",
        f"Features: {len(FEATURES)}",
        f"Nonzero certificate weights: {len(certificate)}",
        f"Sum lambda: {certificate['lambda_full_precision'].sum():.16g}",
        f"UPA lambda mass: {float(result['positive_mass']):.16g}",
        f"BPA lambda mass: {float(result['negative_mass']):.16g}",
        f"max abs(U.T @ lambda): {float(result['certificate_residual']):.16g}",
        f"max convex-hull combination difference: {float(result['hull_residual']):.16g}",
        "Conclusion: the standardized UPA and BPA convex hulls intersect,",
        "so the full dataset is not linearly separable.",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")

    fig, (ax_values, ax_weights) = plt.subplots(
        1, 2, figsize=(10.8, 4.6), constrained_layout=True
    )
    positions = np.arange(len(FEATURES))
    ax_values.axhline(0.0, color="#9CA3AF", linewidth=0.8)
    ax_values.plot(
        positions,
        positive_combination,
        color="#1F77B4",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.7,
        linewidth=1.6,
        label="UPA convex combination",
        zorder=3,
    )
    ax_values.plot(
        positions,
        negative_combination,
        color="#F28E2B",
        marker="x",
        markersize=7,
        markeredgewidth=1.7,
        linestyle="--",
        linewidth=1.3,
        label="BPA convex combination",
        zorder=4,
    )
    ax_values.set_xticks(positions, FEATURES, rotation=42, ha="right")
    ax_values.set_ylabel("Standardized feature value")
    ax_values.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_values.set_title("(a) Equal points in the two convex hulls")
    ax_values.grid(axis="y", alpha=0.2)
    ax_values.legend(fontsize=8.5, loc="upper right")
    residual = float(result["hull_residual"])
    exponent = int(np.floor(np.log10(residual)))
    mantissa = residual / (10.0**exponent)
    ax_values.text(
        0.02,
        0.03,
        rf"maximum coordinate difference $={mantissa:.2f}\times10^{{{exponent}}}$",
        transform=ax_values.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    colors = np.where(certificate["class"].eq("UPA"), "#1F77B4", "#F28E2B")
    y_positions = np.arange(len(certificate))
    ax_weights.barh(
        y_positions,
        certificate["lambda_full_precision"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax_weights.set_yticks(y_positions, certificate["PatientID"])
    ax_weights.invert_yaxis()
    ax_weights.set_xlabel(r"Farkas weight $\lambda_i$")
    ax_weights.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_weights.set_title("(b) Twelve nonzero certificate weights")
    ax_weights.grid(axis="x", alpha=0.2)
    ax_weights.text(
        0.98,
        0.02,
        "blue: UPA   orange: BPA",
        transform=ax_weights.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    fig.suptitle(
        "Full-data certificate of linear hard-margin infeasibility",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    save_outputs(compute_certificate())


if __name__ == "__main__":
    main()

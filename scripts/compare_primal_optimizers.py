#!/usr/bin/env python3
"""Compare stochastic subgradient and momentum variants on the SVM primal.

All methods use the same selected-C, regularized hinge-loss objective, zero
initialization, sampled-observation sequence, and iteration budget.  Heavy Ball
and Nesterov are deliberately described as momentum *subgradient* variants:
the classical acceleration guarantees for smooth objectives do not apply at
the hinge-loss kink.

Writes:
    output/csv/primal_optimizer_comparison.csv
    output/tables/primal_optimizer_comparison_table.tex
    output/figures/primal_optimizer_comparison.{png,pdf}
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

import _pubstyle
from linear_hard_margin_svm import FEATURES, LOG_FEATURES

_pubstyle.apply()

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cgh_pa_dataset.csv"
CSV_PATH = ROOT / "output" / "csv" / "primal_optimizer_comparison.csv"
TABLE_PATH = ROOT / "output" / "tables" / "primal_optimizer_comparison_table.tex"
SLIDE_TABLE_PATH = ROOT / "output" / "tables" / "primal_optimizer_slide_table.tex"
FIGURE_STEM = ROOT / "output" / "figures" / "primal_optimizer_comparison"

SELECTED_C = 1e-2
EXPECTED_OBSERVATIONS = 114
LAMBDA = 1.0 / (EXPECTED_OBSERVATIONS * SELECTED_C)
STEPS = 50_000
SEED = 2026
STEP_OFFSET = 100
ETA_SCALES = (0.25, 0.5, 1.0, 2.0)
MOMENTA = (0.3, 0.5, 0.7, 0.9)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Return standardized features with a constant feature and signs in {-1,+1}."""
    frame = pd.read_csv(DATA_PATH)
    labels = np.where(frame["Model 1 Target"].eq("UPA"), 1.0, -1.0)
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    for column in LOG_FEATURES:
        features[column] = np.log(features[column] + 1e-6)
    features = features.fillna(features.median())
    features = (features - features.mean()) / features.std(ddof=0)
    design = np.column_stack([features.to_numpy(dtype=float), np.ones(len(frame))])
    return design, labels


def objective(theta: np.ndarray, design: np.ndarray, labels: np.ndarray) -> float:
    margins = labels * (design @ theta)
    return float(
        np.maximum(0.0, 1.0 - margins).mean()
        + 0.5 * LAMBDA * (theta[:-1] @ theta[:-1])
    )


def reference_solution(
    design: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, float]:
    """Solve the equivalent slack-variable convex quadratic program."""
    observations, dimensions = design.shape

    def qp_objective(values: np.ndarray) -> float:
        weights = values[: dimensions - 1]
        slack = values[dimensions:]
        return float(0.5 * LAMBDA * (weights @ weights) + slack.mean())

    def qp_gradient(values: np.ndarray) -> np.ndarray:
        gradient = np.empty_like(values)
        gradient[: dimensions - 1] = LAMBDA * values[: dimensions - 1]
        gradient[dimensions - 1] = 0.0
        gradient[dimensions:] = 1.0 / observations
        return gradient

    signed_design = labels[:, None] * design
    constraint_jacobian = np.column_stack(
        [signed_design, np.eye(observations)]
    )
    constraints = {
        "type": "ineq",
        "fun": lambda values: (
            signed_design @ values[:dimensions]
            + values[dimensions:]
            - 1.0
        ),
        "jac": lambda values: constraint_jacobian,
    }
    initial = np.concatenate([np.zeros(dimensions), np.ones(observations)])
    bounds = [(None, None)] * dimensions + [(0.0, None)] * observations
    result = minimize(
        qp_objective,
        initial,
        jac=qp_gradient,
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 10_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Reference QP failed: {result.message}")
    theta = result.x[:dimensions]
    return theta, objective(theta, design, labels)


def stochastic_subgradient(
    theta: np.ndarray, row: np.ndarray, label: float
) -> np.ndarray:
    grad = LAMBDA * theta
    grad[-1] = 0.0
    if label * float(row @ theta) < 1.0:
        grad = grad - label * row
    return grad


def run_method(
    method: str,
    design: np.ndarray,
    labels: np.ndarray,
    samples: np.ndarray,
    eta_scale: float,
    beta: float,
) -> dict[str, object]:
    """Run one method and retain objective histories for raw and averaged iterates."""
    theta = np.zeros(design.shape[1])
    previous = theta.copy()
    average = np.zeros_like(theta)
    raw_objectives = np.empty(STEPS)
    average_objectives = np.empty(STEPS)

    for step, sample in enumerate(samples, start=1):
        eta = eta_scale / (LAMBDA * (step + STEP_OFFSET))
        row = design[sample]
        label = labels[sample]

        if method == "SSG":
            updated = theta - eta * stochastic_subgradient(theta, row, label)
        elif method == "Heavy Ball":
            updated = (
                theta
                - eta * stochastic_subgradient(theta, row, label)
                + beta * (theta - previous)
            )
        elif method == "Nesterov-style":
            lookahead = theta + beta * (theta - previous)
            updated = lookahead - eta * stochastic_subgradient(
                lookahead, row, label
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        previous, theta = theta, updated
        average += (theta - average) / step
        raw_objectives[step - 1] = objective(theta, design, labels)
        average_objectives[step - 1] = objective(average, design, labels)

    return {
        "theta": theta,
        "average": average,
        "raw_objectives": raw_objectives,
        "average_objectives": average_objectives,
    }


def first_threshold(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(np.minimum.accumulate(values) <= threshold)
    return int(indices[0] + 1) if len(indices) else None


def display_steps(value: int | None) -> str:
    return "---" if value is None else f"{value:,}"


def main() -> None:
    design, labels = load_data()
    if len(labels) != EXPECTED_OBSERVATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_OBSERVATIONS} observations, found {len(labels)}; "
            "recompute LAMBDA from the selected C."
        )
    reference_theta, optimum = reference_solution(design, labels)
    samples = np.random.default_rng(SEED).integers(0, len(labels), size=STEPS)

    candidates: dict[str, list[dict[str, object]]] = {
        "SSG": [],
        "Heavy Ball": [],
        "Nesterov-style": [],
    }
    for method in candidates:
        beta_grid = (0.0,) if method == "SSG" else MOMENTA
        for beta in beta_grid:
            for eta_scale in ETA_SCALES:
                result = run_method(
                    method, design, labels, samples, eta_scale, beta
                )
                result["eta_scale"] = eta_scale
                result["beta"] = beta
                candidates[method].append(result)

    selected = {
        method: min(runs, key=lambda run: run["average_objectives"][-1])
        for method, runs in candidates.items()
    }

    rows: list[dict[str, object]] = []
    for method, result in selected.items():
        averaged = result["average"]
        raw_objectives = result["raw_objectives"]
        average_objectives = result["average_objectives"]
        scores = design @ averaged
        cosine = float(
            averaged @ reference_theta
            / (np.linalg.norm(averaged) * np.linalg.norm(reference_theta))
        )
        raw_gaps = raw_objectives - optimum
        five_steps = first_threshold(raw_gaps, 0.05 * optimum)
        one_steps = first_threshold(raw_gaps, 0.01 * optimum)
        rows.append(
            {
                "method": method,
                "C": SELECTED_C,
                "lambda": LAMBDA,
                "eta_scale": float(result["eta_scale"]),
                "beta": float(result["beta"]),
                "final_averaged_objective": float(average_objectives[-1]),
                "final_averaged_gap": float(average_objectives[-1] - optimum),
                "best_raw_gap": float(np.min(raw_objectives - optimum)),
                "steps_to_5pct": five_steps,
                "epochs_to_5pct": None if five_steps is None else five_steps / len(labels),
                "steps_to_1pct": one_steps,
                "epochs_to_1pct": None if one_steps is None else one_steps / len(labels),
                "averaged_iterate_AUROC": float(roc_auc_score(labels > 0, scores)),
                "cosine_to_reference": cosine,
            }
        )

    comparison = pd.DataFrame(rows)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(CSV_PATH, index=False)

    table_rows = []
    for row in comparison.itertuples(index=False):
        method_label = {
            "SSG": "SSG",
            "Heavy Ball": "Heavy Ball",
            "Nesterov-style": "Nesterov-style",
        }[row.method]
        beta = "---" if row.method == "SSG" else f"{row.beta:.1f}"
        table_rows.append(
            f"{method_label} & {row.eta_scale:.2f} & {beta} & "
            f"{row.final_averaged_objective:.4f} & "
            f"{display_steps(row.steps_to_5pct)} & "
            f"{display_steps(row.steps_to_1pct)} & "
            f"{row.averaged_iterate_AUROC:.3f} & "
            f"{row.cosine_to_reference:.3f} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Method & $a$ & $\beta$ & $P(\bar w_K)$ & 5\% steps & 1\% steps & AUROC & Cosine \\",
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(table, encoding="utf-8")

    slide_rows = []
    for row in comparison.itertuples(index=False):
        slide_rows.append(
            f"{row.method} & {display_steps(row.steps_to_5pct)} & "
            f"{display_steps(row.steps_to_1pct)} & "
            f"{row.final_averaged_objective:.4f} & "
            f"{row.averaged_iterate_AUROC:.3f} \\\\"
        )
    slide_table = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Method & 5\% steps & 1\% steps & $P(\bar w_K)$ & AUROC \\",
            r"\midrule",
            *slide_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    SLIDE_TABLE_PATH.write_text(slide_table, encoding="utf-8")

    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_objective, ax_gap) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    steps = np.arange(1, STEPS + 1)
    colors = {"SSG": "#0072B2", "Heavy Ball": "#D55E00", "Nesterov-style": "#009E73"}
    for method, result in selected.items():
        color = colors[method]
        averaged_objectives = result["average_objectives"]
        best_gap = np.maximum(
            np.minimum.accumulate(result["raw_objectives"]) - optimum, 1e-10
        )
        ax_objective.semilogx(
            steps, averaged_objectives, color=color, linewidth=1.7, label=method
        )
        ax_gap.loglog(steps, best_gap, color=color, linewidth=1.7, label=method)

    ax_objective.axhline(optimum, color="black", linestyle="--", linewidth=1.0,
                         label=r"reference $P^\star$")
    ax_objective.set_xlabel("Stochastic update")
    ax_objective.set_ylabel(r"Averaged-iterate objective $P(\bar w_k)$")
    ax_objective.set_title(r"Objective convergence at $C=0.01$")
    ax_objective.legend(fontsize=8)

    ax_gap.axhline(0.05 * optimum, color="0.35", linestyle="--", linewidth=1.0,
                   label=r"5\% of $P^\star$")
    ax_gap.axhline(0.01 * optimum, color="0.55", linestyle=":", linewidth=1.0,
                   label=r"1\% of $P^\star$")
    ax_gap.set_xlabel("Stochastic update")
    ax_gap.set_ylabel(r"Best raw-iterate gap $P(w_k)-P^\star$")
    ax_gap.set_title("Best objective gap")
    ax_gap.legend(fontsize=8)
    for axis in (ax_objective, ax_gap):
        axis.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"))
    plt.close(fig)

    print(f"Selected C: {SELECTED_C:.6g}  lambda=1/(mC): {LAMBDA:.12f}")
    print(f"Reference normalized objective: {optimum:.12f}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

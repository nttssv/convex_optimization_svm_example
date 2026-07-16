#!/usr/bin/env python3
"""Feasibility-first evaluation of the linear hard-margin SVM.

This script implements the constrained primal problem

    minimize    0.5 * ||w||^2
    subject to  t_i (x_i^T w + b) >= 1  for every training sample i.

There are no slack variables, finite-C penalties, or kernels.  Each
cross-validation training fold is tested for linear separability before the
quadratic program is solved.  A ROC curve is produced only when every fold is
feasible; otherwise the script records that no valid hard-margin model or ROC
exists for the requested formulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


DATA_PATH = Path("data/cgh_pa_dataset.csv")
OUT_DIR = Path("output/figures")
CSV_DIR = Path("output/csv")
TABLE_DIR = Path("output/tables")
FEATURES = [
    "PAC",
    "PRA",
    "Potassium",
    "Tumor size",
    "18-OHF",
    "18-oxoF",
    "Systolic BP",
    "Diastolic BP",
    "DDD",
    "Age",
]
LOG_FEATURES = ["PAC", "PRA", "18-OHF", "18-oxoF"]


@dataclass
class Preprocessor:
    """Fold-specific imputation and standardization parameters."""

    medians: pd.Series
    means: pd.Series
    scales: pd.Series

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> tuple["Preprocessor", np.ndarray]:
        transformed = _transform_hormones(frame)
        medians = transformed.median()
        complete = transformed.fillna(medians)
        means = complete.mean()
        scales = complete.std(ddof=0).replace(0.0, 1.0)
        fitted = cls(medians=medians, means=means, scales=scales)
        return fitted, fitted.transform(frame)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = _transform_hormones(frame).fillna(self.medians)
        standardized = (transformed - self.means) / self.scales
        return standardized.to_numpy(dtype=float)


def _transform_hormones(frame: pd.DataFrame) -> pd.DataFrame:
    transformed = frame.copy()
    for column in LOG_FEATURES:
        transformed[column] = np.log(transformed[column] + 1e-6)
    return transformed


def signed_design(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return rows t_i [x_i^T, 1] used by the margin constraints."""
    return labels[:, None] * np.column_stack([X, np.ones(len(X))])


def feasibility_point(
    X: np.ndarray, labels: np.ndarray
) -> tuple[bool, np.ndarray | None, str]:
    """Find any theta satisfying t_i(x_i^T w+b) >= 1."""
    design = signed_design(X, labels)
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-design,
        b_ub=-np.ones(len(X)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    point = result.x if result.success else None
    return bool(result.success), point, str(result.message)


def objective(theta: np.ndarray) -> float:
    """Primal objective phi(theta)=0.5||w||^2; b is unpenalized."""
    return float(0.5 * theta[:-1] @ theta[:-1])


def objective_gradient(theta: np.ndarray) -> np.ndarray:
    """First derivative [w^T, 0]^T of the primal objective."""
    return np.r_[theta[:-1], 0.0]


def objective_hessian(theta: np.ndarray) -> np.ndarray:
    """Second derivative diag(I_p, 0) of the primal objective."""
    del theta
    hessian = np.eye(len(FEATURES) + 1)
    hessian[-1, -1] = 0.0
    return hessian


def fit_primal_hard_margin(
    X: np.ndarray, labels: np.ndarray, initial_theta: np.ndarray
) -> np.ndarray:
    """Solve the feasible hard-margin primal quadratic program."""
    design = signed_design(X, labels)
    constraint = {
        "type": "ineq",
        "fun": lambda theta: design @ theta - 1.0,
        "jac": lambda theta: design,
    }
    result = minimize(
        objective,
        initial_theta,
        jac=objective_gradient,
        constraints=constraint,
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 20_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Hard-margin primal solver failed: {result.message}")
    minimum_margin = float(np.min(design @ result.x))
    if minimum_margin < 1.0 - 1e-6:
        raise RuntimeError(
            f"Returned parameters violate the margin: minimum={minimum_margin:.8f}."
        )
    return result.x


def save_roc(y: np.ndarray, scores: np.ndarray, output_path: Path) -> float:
    """Save the pooled out-of-fold ROC if all hard-margin fits exist."""
    fpr, tpr, _ = roc_curve(y, scores)
    auroc = float(roc_auc_score(y, scores))
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.plot(fpr, tpr, color="#1f5a92", linewidth=2.4, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0, label="Chance")
    ax.set(
        xlabel="False-positive rate (1 - specificity)",
        ylabel="True-positive rate (sensitivity)",
        title="Linear hard-margin SVM: five-fold out-of-fold ROC",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return auroc


def save_feasibility_audit(
    full_feasible: bool, results: pd.DataFrame
) -> pd.DataFrame:
    """Save the full-data/fold audit to CSV and a LaTeX table fragment."""
    audit_rows: list[dict[str, object]] = [
        {
            "data_split": "Full dataset",
            "n_train": 114,
            "n_held_out": "--",
            "linear_hard_margin_feasible": full_feasible,
        }
    ]
    for row in results.itertuples(index=False):
        audit_rows.append(
            {
                "data_split": f"Fold {row.fold}",
                "n_train": int(row.n_train),
                "n_held_out": int(row.n_test),
                "linear_hard_margin_feasible": bool(
                    row.linear_hard_margin_feasible
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(CSV_DIR / "linear_hard_margin_feasibility.csv", index=False)

    latex_rows = []
    for row in audit.itertuples(index=False):
        status = "Yes" if row.linear_hard_margin_feasible else "No"
        latex_rows.append(
            f"{row.data_split} & {row.n_train} & {row.n_held_out} & {status} \\\\"
        )
    table = "\n".join(
        [
            r"\begin{tabular}{lrrc}",
            r"\toprule",
            r"Data split & Training observations & Held-out observations & Feasible \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (TABLE_DIR / "linear_hard_margin_feasibility_table.tex").write_text(
        table, encoding="utf-8"
    )
    return audit


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    X_raw = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = (frame["Model 1 Target"] == "UPA").astype(int).to_numpy()
    labels = 2.0 * y - 1.0

    _, X_full = Preprocessor.fit(X_raw)
    full_feasible, _, full_message = feasibility_point(X_full, labels)

    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = np.full(len(y), np.nan)
    rows: list[dict[str, object]] = []

    for fold, (train_indices, test_indices) in enumerate(
        cross_validator.split(X_raw, y), start=1
    ):
        preprocessor, X_train = Preprocessor.fit(X_raw.iloc[train_indices])
        X_test = preprocessor.transform(X_raw.iloc[test_indices])
        train_labels = labels[train_indices]
        feasible, initial_theta, message = feasibility_point(X_train, train_labels)
        row: dict[str, object] = {
            "fold": fold,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "linear_hard_margin_feasible": feasible,
            "solver_message": message,
        }
        if feasible and initial_theta is not None:
            theta = fit_primal_hard_margin(X_train, train_labels, initial_theta)
            scores[test_indices] = X_test @ theta[:-1] + theta[-1]
            row["minimum_training_margin"] = float(
                np.min(signed_design(X_train, train_labels) @ theta)
            )
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "linear_hard_margin_feasibility.csv", index=False)
    save_feasibility_audit(full_feasible, results)
    all_folds_feasible = bool(results["linear_hard_margin_feasible"].all())

    summary = [
        "Linear hard-margin SVM feasibility audit",
        "========================================",
        f"Data: {DATA_PATH}",
        f"Samples: {len(y)} (UPA={int(y.sum())}, BPA={int(len(y)-y.sum())})",
        f"Features: {len(FEATURES)}",
        f"Full dataset feasible: {full_feasible}",
        f"Full-data solver message: {full_message}",
        f"All five training folds feasible: {all_folds_feasible}",
    ]
    if all_folds_feasible:
        auroc = save_roc(y, scores, OUT_DIR / "linear_hard_margin_roc.png")
        summary.append(f"Pooled out-of-fold AUROC: {auroc:.6f}")
    else:
        summary.append(
            "ROC not computed: at least one training fold has no feasible linear hard-margin solution."
        )

    (OUT_DIR / "linear_hard_margin_summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    print("\nPer-fold feasibility")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

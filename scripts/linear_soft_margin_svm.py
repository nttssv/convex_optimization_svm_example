#!/usr/bin/env python3
"""Tune and audit a linear soft-margin SVM on the project dataset.

The implementation uses a leakage-safe pipeline and nested cross-validation.
The inner loop selects C by ROC-AUC; the outer loop estimates performance.
Afterwards, a full-data grid search selects the reported C and fits one final
linear model for primal/dual/slack diagnostics. All numerical artifacts are
written to CSV before LaTeX table fragments and figures are produced.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import _pubstyle; _pubstyle.apply()
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from linear_hard_margin_svm import FEATURES, LOG_FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cgh_pa_dataset.csv"
CSV_DIR = ROOT / "output" / "csv"
TABLE_DIR = ROOT / "output" / "tables"
FIGURE_STEM = ROOT / "output" / "figures" / "linear_soft_margin_grid_roc"

# A half-log10 grid covers strong through weak regularization while keeping the
# smallest reported value distinguishable at the requested two decimals.
C_GRID = 10.0 ** np.arange(-2.0, 2.01, 0.5)
RANDOM_STATE = 0

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def log_selected_columns(values: np.ndarray) -> np.ndarray:
    """Apply the report's log transform without learning from held-out data."""
    transformed = np.asarray(values, dtype=float).copy()
    for column in LOG_FEATURES:
        index = FEATURES.index(column)
        transformed[:, index] = np.log(transformed[:, index] + 1e-6)
    return transformed


def make_pipeline() -> Pipeline:
    """Return the leakage-safe linear soft-margin SVM pipeline."""
    return Pipeline(
        [
            (
                "log_transform",
                FunctionTransformer(log_selected_columns, validate=False),
            ),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
            ("svc", SVC(kernel="linear", C=1.0, tol=1e-7, max_iter=-1)),
        ]
    )


def nested_grid_search(
    features: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Select C inside each outer fold and return unbiased OOF scores."""
    outer_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    oof_scores = np.full(len(labels), np.nan)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(features, labels), start=1
    ):
        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold
        )
        search = GridSearchCV(
            make_pipeline(),
            {"svc__C": C_GRID},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(features.iloc[train_indices], labels[train_indices])
        fold_scores = search.decision_function(features.iloc[test_indices])
        oof_scores[test_indices] = fold_scores
        fold_rows.append(
            {
                "outer_fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "best_C_full_precision": float(search.best_params_["svc__C"]),
                "best_C_display_2dp": f"{float(search.best_params_['svc__C']):.2f}",
                "inner_best_mean_AUROC": float(search.best_score_),
                "outer_test_AUROC": float(
                    roc_auc_score(labels[test_indices], fold_scores)
                ),
            }
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("Nested cross-validation did not score every observation.")
    predictions = (oof_scores >= 0.0).astype(int)
    prediction_table = pd.DataFrame(
        {
            "observation_number": np.arange(1, len(labels) + 1),
            "true_label": labels,
            "oof_decision_score_full_precision": oof_scores,
            "oof_decision_score_display_2dp": [f"{value:.2f}" for value in oof_scores],
            "predicted_label_at_zero": predictions,
        }
    )
    return pd.DataFrame(fold_rows), prediction_table, oof_scores


def fit_full_grid(
    features: pd.DataFrame, labels: np.ndarray
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Run the reported full-data grid search and return its validation curve."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(),
        {"svc__C": C_GRID},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(features, labels)
    results = pd.DataFrame(search.cv_results_)
    grid_table = pd.DataFrame(
        {
            "C_full_precision": results["param_svc__C"].astype(float),
            "C_display_2dp": [f"{float(value):.2f}" for value in results["param_svc__C"]],
            "mean_train_AUROC": results["mean_train_score"],
            "std_train_AUROC": results["std_train_score"],
            "mean_validation_AUROC": results["mean_test_score"],
            "std_validation_AUROC": results["std_test_score"],
            "rank_validation_AUROC": results["rank_test_score"].astype(int),
        }
    ).sort_values("C_full_precision")
    return search, grid_table


def audit_final_model(
    search: GridSearchCV,
    features: pd.DataFrame,
    labels: np.ndarray,
    patient_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute primal, dual, KKT, slack, and support-vector diagnostics."""
    best = search.best_estimator_
    transformed = best[:-1].transform(features)
    svc = best.named_steps["svc"]
    weights = svc.coef_[0]
    intercept = float(svc.intercept_[0])
    scores = transformed @ weights + intercept
    signed_margins = (2.0 * labels - 1.0) * scores
    slacks = np.maximum(0.0, 1.0 - signed_margins)
    best_c = float(search.best_params_["svc__C"])

    support_indices = svc.support_
    support_alpha = np.abs(svc.dual_coef_[0])
    support_signed_alpha = svc.dual_coef_[0]
    alpha = np.zeros(len(labels))
    alpha[support_indices] = support_alpha
    signed_labels = 2.0 * labels - 1.0

    primal_objective = float(0.5 * weights @ weights + best_c * slacks.sum())
    dual_objective = float(alpha.sum() - 0.5 * weights @ weights)
    reconstruction_residual = float(
        np.max(
            np.abs(
                weights
                - support_signed_alpha @ transformed[support_indices]
            )
        )
    )
    equality_residual = float(abs(alpha @ signed_labels))
    primal_constraint_residual = signed_margins - 1.0 + slacks
    margin_complementarity_residual = float(
        np.max(np.abs(alpha * (1.0 - slacks - signed_margins)))
    )
    slack_complementarity_residual = float(
        np.max(np.abs((best_c - alpha) * slacks))
    )

    if float(np.min(primal_constraint_residual)) < -1e-7:
        raise RuntimeError("Final model violates a soft-margin primal constraint.")
    if float(np.max(alpha)) > best_c + 1e-7:
        raise RuntimeError("Final model violates the dual upper bound alpha <= C.")
    if primal_objective - dual_objective > 1e-6:
        raise RuntimeError("Final primal-dual gap exceeds numerical tolerance.")

    support_table = pd.DataFrame(
        {
            "observation_number": support_indices + 1,
            "PatientID": patient_ids.iloc[support_indices].to_numpy(),
            "class": np.where(labels[support_indices] == 1, "UPA", "BPA"),
            "alpha_full_precision": support_alpha,
            "alpha_display_2dp": [f"{value:.2f}" for value in support_alpha],
            "decision_score_full_precision": scores[support_indices],
            "decision_score_display_2dp": [f"{value:.2f}" for value in scores[support_indices]],
            "signed_margin_full_precision": signed_margins[support_indices],
            "signed_margin_display_2dp": [f"{value:.2f}" for value in signed_margins[support_indices]],
            "slack_full_precision": slacks[support_indices],
            "slack_display_2dp": [f"{value:.2f}" for value in slacks[support_indices]],
            "at_upper_bound": np.isclose(support_alpha, best_c, atol=1e-5),
        }
    )

    audit_rows = [
        ("best_C", best_c, f"{best_c:.2f}"),
        ("full_grid_best_mean_validation_AUROC", float(search.best_score_), f"{search.best_score_:.2f}"),
        ("number_of_support_vectors", float(len(support_indices)), f"{len(support_indices)}"),
        ("number_with_positive_slack", float(np.count_nonzero(slacks > 1e-7)), f"{np.count_nonzero(slacks > 1e-7)}"),
        ("sum_slack", float(slacks.sum()), f"{slacks.sum():.2f}"),
        ("weight_squared_norm", float(weights @ weights), f"{weights @ weights:.2f}"),
        ("intercept", intercept, f"{intercept:.2f}"),
        ("primal_objective", primal_objective, f"{primal_objective:.2f}"),
        ("dual_objective", dual_objective, f"{dual_objective:.2f}"),
        ("primal_dual_gap", primal_objective - dual_objective, f"{primal_objective - dual_objective:.2e}"),
        ("dual_equality_residual", equality_residual, f"{equality_residual:.2e}"),
        ("weight_reconstruction_residual", reconstruction_residual, f"{reconstruction_residual:.2e}"),
        ("margin_complementarity_residual", margin_complementarity_residual, f"{margin_complementarity_residual:.2e}"),
        ("slack_complementarity_residual", slack_complementarity_residual, f"{slack_complementarity_residual:.2e}"),
        ("minimum_primal_constraint_residual", float(np.min(primal_constraint_residual)), f"{float(np.min(primal_constraint_residual)):.2e}"),
    ]
    audit = pd.DataFrame(
        audit_rows,
        columns=["metric", "full_precision_value", "display_value"],
    )
    coefficient_table = pd.DataFrame(
        {
            "parameter": [*FEATURES, "Intercept"],
            "coefficient_full_precision": [*weights, intercept],
            "coefficient_display_2dp": [
                *[f"{value:.2f}" for value in weights],
                f"{intercept:.2f}",
            ],
        }
    )
    return audit, support_table, coefficient_table


def save_latex_tables(
    nested: pd.DataFrame,
    audit: pd.DataFrame,
    coefficients: pd.DataFrame,
    nested_metrics: dict[str, float],
) -> None:
    """Create report tables from the same values written to CSV."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metric = dict(zip(audit["metric"], audit["display_value"]))
    summary_rows = [
        ("Selected $C$ from full-data grid search", metric["best_C"]),
        ("Mean inner-CV AUROC at selected $C$", metric["full_grid_best_mean_validation_AUROC"]),
        ("Nested out-of-fold AUROC", f"{nested_metrics['oof_auc']:.2f}"),
        ("Nested out-of-fold accuracy", f"{nested_metrics['accuracy']:.2f}"),
        ("Nested sensitivity", f"{nested_metrics['sensitivity']:.2f}"),
        ("Nested specificity", f"{nested_metrics['specificity']:.2f}"),
        ("Full-data support vectors", metric["number_of_support_vectors"]),
        (r"Full-data observations with $\xi_i>0$", metric["number_with_positive_slack"]),
        (r"Full-data $\sum_i\xi_i$", metric["sum_slack"]),
        ("Full-data primal objective", metric["primal_objective"]),
        ("Full-data dual objective", metric["dual_objective"]),
    ]
    summary_table = "\n".join(
        [r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
        + [f"{name} & {value} \\\\" for name, value in summary_rows]
        + [r"\bottomrule", r"\end{tabular}", ""]
    )
    (TABLE_DIR / "soft_margin_summary_table.tex").write_text(
        summary_table, encoding="utf-8"
    )

    fold_rows = []
    for row in nested.itertuples(index=False):
        fold_rows.append(
            f"{row.outer_fold} & {row.n_train} & {row.n_test} & "
            f"{float(row.best_C_full_precision):.2f} & "
            f"{row.inner_best_mean_AUROC:.2f} & {row.outer_test_AUROC:.2f} \\\\"
        )
    fold_table = "\n".join(
        [
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Outer fold & Train & Test & Selected $C$ & Inner AUROC & Outer AUROC \\",
            r"\midrule",
            *fold_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (TABLE_DIR / "soft_margin_nested_cv_table.tex").write_text(
        fold_table, encoding="utf-8"
    )

    coefficient_rows = []
    for left_index in range(5):
        left = coefficients.iloc[left_index]
        right = coefficients.iloc[left_index + 5]
        coefficient_rows.append(
            f"{left['parameter']} & {left['coefficient_display_2dp']} & "
            f"{right['parameter']} & {right['coefficient_display_2dp']} \\\\"
        )
    intercept = coefficients.iloc[-1]
    coefficient_rows.append(
        f"{intercept['parameter']} & {intercept['coefficient_display_2dp']} & & \\\\"
    )
    coefficient_table = "\n".join(
        [
            r"\begin{tabular}{lr@{\qquad}lr}",
            r"\toprule",
            r"Parameter & Estimate & Parameter & Estimate \\",
            r"\midrule",
            *coefficient_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (TABLE_DIR / "soft_margin_coefficients_table.tex").write_text(
        coefficient_table, encoding="utf-8"
    )


def save_figure(
    grid: pd.DataFrame,
    labels: np.ndarray,
    oof_scores: np.ndarray,
    best_c: float,
    oof_auc: float,
) -> None:
    """Save the C validation curve and nested out-of-fold ROC."""
    FIGURE_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_grid, ax_roc) = plt.subplots(
        1, 2, figsize=(10.6, 4.4), constrained_layout=True
    )
    x = grid["C_full_precision"].to_numpy()
    mean = grid["mean_validation_AUROC"].to_numpy()
    std = grid["std_validation_AUROC"].to_numpy()
    ax_grid.semilogx(x, mean, color="#1F77B4", marker="o", linewidth=1.8)
    ax_grid.fill_between(x, mean - std, mean + std, color="#1F77B4", alpha=0.18)
    ax_grid.axvline(best_c, color="#D62728", linestyle="--", linewidth=1.4)
    ax_grid.set(
        xlabel="Penalty parameter $C$ (log scale)",
        ylabel="Mean five-fold validation AUROC",
        title="(a) Grid search for $C$",
        ylim=(0.45, 1.01),
    )
    ax_grid.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_grid.grid(alpha=0.22)
    ax_grid.text(
        0.97,
        0.05,
        f"selected $C={best_c:.2f}$",
        transform=ax_grid.transAxes,
        ha="right",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    fpr, tpr, _ = roc_curve(labels, oof_scores)
    ax_roc.plot(fpr, tpr, color="#1F77B4", linewidth=2.0, label=f"AUROC = {oof_auc:.2f}")
    ax_roc.plot([0, 1], [0, 1], color="#6B7280", linestyle=":", linewidth=1.2, label="Chance")
    ax_roc.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="(b) Nested out-of-fold ROC",
        xlim=(-0.01, 1.01),
        ylim=(-0.01, 1.01),
    )
    ax_roc.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_roc.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_roc.set_aspect("equal", adjustable="box")
    ax_roc.grid(alpha=0.22)
    ax_roc.legend(loc="lower right")
    fig.suptitle("Linear soft-margin SVM: model selection and validation", fontweight="bold")
    fig.savefig(FIGURE_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(FIGURE_STEM.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = frame["Model 1 Target"].eq("UPA").astype(int).to_numpy()

    nested, predictions, oof_scores = nested_grid_search(features, labels)
    search, grid = fit_full_grid(features, labels)
    audit, support, coefficients = audit_final_model(
        search, features, labels, frame["PatientID"]
    )

    tn, fp, fn, tp = confusion_matrix(labels, oof_scores >= 0.0).ravel()
    nested_metrics = {
        "oof_auc": float(roc_auc_score(labels, oof_scores)),
        "accuracy": float(accuracy_score(labels, oof_scores >= 0.0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
    }

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    nested.to_csv(CSV_DIR / "soft_margin_nested_cv.csv", index=False, float_format="%.15g")
    predictions.insert(1, "PatientID", frame["PatientID"])
    predictions.to_csv(CSV_DIR / "soft_margin_oof_predictions.csv", index=False, float_format="%.15g")
    grid.to_csv(CSV_DIR / "soft_margin_grid_search.csv", index=False, float_format="%.15g")
    audit.to_csv(CSV_DIR / "soft_margin_solution_audit.csv", index=False, float_format="%.15g")
    support.to_csv(CSV_DIR / "soft_margin_support_vectors.csv", index=False, float_format="%.15g")
    coefficients.to_csv(
        CSV_DIR / "soft_margin_coefficients.csv", index=False, float_format="%.15g"
    )
    pd.DataFrame([nested_metrics]).to_csv(
        CSV_DIR / "soft_margin_nested_metrics.csv", index=False, float_format="%.15g"
    )

    save_latex_tables(nested, audit, coefficients, nested_metrics)
    save_figure(
        grid,
        labels,
        oof_scores,
        float(search.best_params_["svc__C"]),
        nested_metrics["oof_auc"],
    )

    print(f"Best full-data C: {float(search.best_params_['svc__C']):.12g}")
    print(f"Full-grid mean validation AUROC: {float(search.best_score_):.6f}")
    print(f"Nested out-of-fold AUROC: {nested_metrics['oof_auc']:.6f}")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()

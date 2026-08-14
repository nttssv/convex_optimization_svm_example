#!/usr/bin/env python3
"""Tune and audit an RBF-kernel soft-margin SVM with nested CV."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import _pubstyle; _pubstyle.apply()
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from linear_hard_margin_svm import FEATURES
from linear_soft_margin_svm import log_selected_columns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cgh_pa_dataset.csv"
CSV_DIR = ROOT / "output" / "csv"
TABLE_DIR = ROOT / "output" / "tables"
FIGURE_DIR = ROOT / "output" / "figures"

# Prespecified primary grid. A separate boundary-sensitivity analysis extends
# both lower bounds after model selection.
C_GRID = 10.0 ** np.arange(-2.0, 2.01, 0.5)
GAMMA_GRID = 10.0 ** np.arange(-3.0, 1.01, 0.5)
RANDOM_STATE = 0

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def display_parameter(value: float) -> str:
    """Use two decimals, switching to scientific notation below 0.01."""
    return f"{value:.2e}" if value < 0.01 else f"{value:.2f}"


def make_pipeline() -> Pipeline:
    """Return a leakage-safe RBF-kernel SVM pipeline."""
    return Pipeline(
        [
            ("log_transform", FunctionTransformer(log_selected_columns, validate=False)),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-7, max_iter=-1)),
        ]
    )


def nested_search(
    features: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune C and gamma inside each outer fold and return OOF predictions."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | int]] = []
    predictions = np.full(len(labels), np.nan)
    fold_assignment = np.zeros(len(labels), dtype=int)

    for fold, (train_indices, test_indices) in enumerate(
        outer.split(features, labels), start=1
    ):
        inner = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE + fold
        )
        search = GridSearchCV(
            make_pipeline(),
            {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID},
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        search.fit(features.iloc[train_indices], labels[train_indices])
        scores = search.decision_function(features.iloc[test_indices])
        predictions[test_indices] = scores
        fold_assignment[test_indices] = fold
        best_index = search.best_index_
        train_auc = float(search.cv_results_["mean_train_score"][best_index])
        validation_auc = float(search.best_score_)
        best_c = float(search.best_params_["svc__C"])
        best_gamma = float(search.best_params_["svc__gamma"])
        rows.append(
            {
                "outer_fold": fold,
                "n_train": len(train_indices),
                "n_test": len(test_indices),
                "best_C_full_precision": best_c,
                "best_C_display_2dp": display_parameter(best_c),
                "best_gamma_full_precision": best_gamma,
                "best_gamma_display_2dp": display_parameter(best_gamma),
                "inner_mean_train_AUROC": train_auc,
                "inner_mean_validation_AUROC": validation_auc,
                "inner_train_validation_gap": train_auc - validation_auc,
                "outer_test_AUROC": float(
                    roc_auc_score(labels[test_indices], scores)
                ),
            }
        )

    if np.isnan(predictions).any() or np.any(fold_assignment == 0):
        raise RuntimeError("Nested RBF search did not score every observation.")
    prediction_table = pd.DataFrame(
        {
            "observation_number": np.arange(1, len(labels) + 1),
            "outer_fold": fold_assignment,
            "true_label": labels,
            "oof_decision_score_full_precision": predictions,
            "oof_decision_score_display_2dp": [f"{value:.2f}" for value in predictions],
            "predicted_label_at_zero": (predictions >= 0.0).astype(int),
        }
    )
    return pd.DataFrame(rows), prediction_table


def full_grid_search(
    features: pd.DataFrame, labels: np.ndarray
) -> tuple[GridSearchCV, pd.DataFrame]:
    """Tune the reported final RBF model on the complete dataset."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(),
        {"svc__C": C_GRID, "svc__gamma": GAMMA_GRID},
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(features, labels)
    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(
        {
            "C_full_precision": results["param_svc__C"].astype(float),
            "C_display_2dp": [display_parameter(float(value)) for value in results["param_svc__C"]],
            "gamma_full_precision": results["param_svc__gamma"].astype(float),
            "gamma_display_2dp": [display_parameter(float(value)) for value in results["param_svc__gamma"]],
            "mean_train_AUROC": results["mean_train_score"],
            "std_train_AUROC": results["std_train_score"],
            "mean_validation_AUROC": results["mean_test_score"],
            "std_validation_AUROC": results["std_test_score"],
            "train_validation_gap": results["mean_train_score"] - results["mean_test_score"],
            "rank_validation_AUROC": results["rank_test_score"].astype(int),
        }
    ).sort_values(["C_full_precision", "gamma_full_precision"])
    return search, table


def audit_model(
    search: GridSearchCV,
    features: pd.DataFrame,
    labels: np.ndarray,
    patient_ids: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit RBF primal/dual values in the induced feature space."""
    best = search.best_estimator_
    transformed = best[:-1].transform(features)
    svc = best.named_steps["svc"]
    support_indices = svc.support_
    signed_labels = 2.0 * labels - 1.0
    scores = best.decision_function(features)
    margins = signed_labels * scores
    slacks = np.maximum(0.0, 1.0 - margins)
    alpha_support = np.abs(svc.dual_coef_[0])
    signed_alpha_support = svc.dual_coef_[0]
    alpha = np.zeros(len(labels))
    alpha[support_indices] = alpha_support
    best_c = float(search.best_params_["svc__C"])
    best_gamma = float(search.best_params_["svc__gamma"])

    kernel_matrix = rbf_kernel(
        transformed[support_indices],
        transformed[support_indices],
        gamma=best_gamma,
    )
    rkhs_norm_squared = float(
        signed_alpha_support @ kernel_matrix @ signed_alpha_support
    )
    primal = float(0.5 * rkhs_norm_squared + best_c * slacks.sum())
    dual = float(alpha_support.sum() - 0.5 * rkhs_norm_squared)
    equality_residual = float(abs(alpha @ signed_labels))
    margin_complementarity = float(
        np.max(np.abs(alpha * (1.0 - slacks - margins)))
    )
    slack_complementarity = float(
        np.max(np.abs((best_c - alpha) * slacks))
    )

    if primal - dual > 1e-5:
        raise RuntimeError("RBF primal-dual gap exceeds numerical tolerance.")
    if alpha.max() > best_c + 1e-7:
        raise RuntimeError("RBF solution violates alpha <= C.")

    best_index = search.best_index_
    train_auc = float(search.cv_results_["mean_train_score"][best_index])
    validation_auc = float(search.best_score_)
    audit = pd.DataFrame(
        [
            ("best_C", best_c, display_parameter(best_c)),
            ("best_gamma", best_gamma, display_parameter(best_gamma)),
            ("full_grid_mean_train_AUROC", train_auc, f"{train_auc:.2f}"),
            ("full_grid_mean_validation_AUROC", validation_auc, f"{validation_auc:.2f}"),
            ("full_grid_train_validation_gap", train_auc - validation_auc, f"{train_auc - validation_auc:.2f}"),
            ("number_of_support_vectors", float(len(support_indices)), str(len(support_indices))),
            ("number_with_positive_slack", float(np.count_nonzero(slacks > 1e-7)), str(np.count_nonzero(slacks > 1e-7))),
            ("sum_slack", float(slacks.sum()), f"{slacks.sum():.2f}"),
            ("rkhs_norm_squared", rkhs_norm_squared, f"{rkhs_norm_squared:.2f}"),
            ("primal_objective", primal, f"{primal:.2f}"),
            ("dual_objective", dual, f"{dual:.2f}"),
            ("primal_dual_gap", primal - dual, f"{primal - dual:.2e}"),
            ("dual_equality_residual", equality_residual, f"{equality_residual:.2e}"),
            ("margin_complementarity_residual", margin_complementarity, f"{margin_complementarity:.2e}"),
            ("slack_complementarity_residual", slack_complementarity, f"{slack_complementarity:.2e}"),
        ],
        columns=["metric", "full_precision_value", "display_value"],
    )
    support = pd.DataFrame(
        {
            "observation_number": support_indices + 1,
            "PatientID": patient_ids.iloc[support_indices].to_numpy(),
            "class": np.where(labels[support_indices] == 1, "UPA", "BPA"),
            "alpha_full_precision": alpha_support,
            "alpha_display_2dp": [f"{value:.2f}" for value in alpha_support],
            "signed_margin_full_precision": margins[support_indices],
            "signed_margin_display_2dp": [f"{value:.2f}" for value in margins[support_indices]],
            "slack_full_precision": slacks[support_indices],
            "slack_display_2dp": [f"{value:.2f}" for value in slacks[support_indices]],
            "at_upper_bound": np.isclose(alpha_support, best_c, atol=1e-5),
        }
    )
    return audit, support


def compute_comparison(
    nested: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare RBF nested performance with the existing linear nested audit."""
    linear_nested = pd.read_csv(CSV_DIR / "soft_margin_nested_cv.csv")
    linear_metrics = pd.read_csv(CSV_DIR / "soft_margin_nested_metrics.csv").iloc[0]
    y = predictions["true_label"].to_numpy()
    scores = predictions["oof_decision_score_full_precision"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, scores >= 0.0).ravel()
    rbf_metrics = {
        "pooled_oof_AUROC": float(roc_auc_score(y, scores)),
        "mean_outer_AUROC": float(nested["outer_test_AUROC"].mean()),
        "std_outer_AUROC": float(nested["outer_test_AUROC"].std(ddof=1)),
        "accuracy": float(accuracy_score(y, scores >= 0.0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
    }
    linear_mean = float(linear_nested["outer_test_AUROC"].mean())
    linear_std = float(linear_nested["outer_test_AUROC"].std(ddof=1))
    comparison = pd.DataFrame(
        [
            {
                "model": "Linear soft margin",
                "tuned_parameters": "C",
                "mean_outer_AUROC": linear_mean,
                "std_outer_AUROC": linear_std,
                "pooled_oof_AUROC": float(linear_metrics["oof_auc"]),
                "accuracy": float(linear_metrics["accuracy"]),
                "sensitivity": float(linear_metrics["sensitivity"]),
                "specificity": float(linear_metrics["specificity"]),
            },
            {
                "model": "RBF kernel",
                "tuned_parameters": "C and gamma",
                "mean_outer_AUROC": rbf_metrics["mean_outer_AUROC"],
                "std_outer_AUROC": rbf_metrics["std_outer_AUROC"],
                "pooled_oof_AUROC": rbf_metrics["pooled_oof_AUROC"],
                "accuracy": rbf_metrics["accuracy"],
                "sensitivity": rbf_metrics["sensitivity"],
                "specificity": rbf_metrics["specificity"],
            },
        ]
    )
    return comparison, rbf_metrics


def save_tables(
    nested: pd.DataFrame,
    audit: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    """Write compact LaTeX tables generated from CSV-bound results."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    nested_rows = []
    for row in nested.itertuples(index=False):
        nested_rows.append(
            f"{row.outer_fold} & {display_parameter(row.best_C_full_precision)} & "
            f"{display_parameter(row.best_gamma_full_precision)} & "
            f"{row.inner_mean_train_AUROC:.2f} & {row.inner_mean_validation_AUROC:.2f} & "
            f"{row.outer_test_AUROC:.2f} \\\\"
        )
    nested_rows.append(
        f"Mean & --- & --- & {nested['inner_mean_train_AUROC'].mean():.3f} & "
        f"{nested['inner_mean_validation_AUROC'].mean():.3f} & "
        f"{nested['outer_test_AUROC'].mean():.3f} \\\\"
    )
    nested_table = "\n".join(
        [
            r"\begin{tabular}{rrrrrr}", r"\toprule",
            r"Fold & $C$ & $\gamma$ & Inner train & Inner validation & Outer AUROC \\",
            r"\midrule", *nested_rows, r"\bottomrule", r"\end{tabular}", "",
        ]
    )
    (TABLE_DIR / "rbf_nested_cv_table.tex").write_text(nested_table, encoding="utf-8")

    metric = dict(zip(audit["metric"], audit["display_value"]))
    rows = [
        ("Selected $C$", metric["best_C"]),
        (r"Selected $\gamma$", metric["best_gamma"]),
        ("Mean training AUROC", metric["full_grid_mean_train_AUROC"]),
        ("Mean validation AUROC", metric["full_grid_mean_validation_AUROC"]),
        ("Train--validation gap", metric["full_grid_train_validation_gap"]),
        ("Largest gap anywhere on grid", metric["maximum_grid_train_validation_gap"]),
        ("Mean nested outer AUROC", metric["mean_nested_outer_AUROC"]),
        ("Support vectors", metric["number_of_support_vectors"]),
        (r"Observations with $\xi_i>0$", metric["number_with_positive_slack"]),
        ("Primal objective", metric["primal_objective"]),
        ("Dual objective", metric["dual_objective"]),
    ]
    summary = "\n".join(
        [r"\begin{tabular}{lr}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
        + [f"{name} & {value} \\\\" for name, value in rows]
        + [r"\bottomrule", r"\end{tabular}", ""]
    )
    (TABLE_DIR / "rbf_summary_table.tex").write_text(summary, encoding="utf-8")

    comparison_rows = []
    for row in comparison.itertuples(index=False):
        comparison_rows.append(
            f"{row.model} & {row.mean_outer_AUROC:.2f} & {row.std_outer_AUROC:.2f} & "
            f"{row.pooled_oof_AUROC:.2f} & {row.accuracy:.2f} & "
            f"{row.sensitivity:.2f} & {row.specificity:.2f} \\\\"
        )
    comparison_table = "\n".join(
        [
            r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Model & Mean outer AUC & SD & Pooled AUC & Accuracy & Sensitivity & Specificity \\",
            r"\midrule", *comparison_rows, r"\bottomrule", r"\end{tabular}", "",
        ]
    )
    (TABLE_DIR / "kernel_comparison_table.tex").write_text(
        comparison_table, encoding="utf-8"
    )


def save_grid_figure(grid: pd.DataFrame, best_c: float, best_gamma: float) -> None:
    """Plot validation AUROC and train-validation gap over C and gamma."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    validation = grid.pivot(index="gamma_full_precision", columns="C_full_precision", values="mean_validation_AUROC")
    gap = grid.pivot(index="gamma_full_precision", columns="C_full_precision", values="train_validation_gap")
    fig, (ax_auc, ax_gap) = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    extent = (-0.5, len(C_GRID) - 0.5, -0.5, len(GAMMA_GRID) - 0.5)
    auc_image = ax_auc.imshow(validation.to_numpy(), origin="lower", aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0, extent=extent)
    gap_image = ax_gap.imshow(gap.to_numpy(), origin="lower", aspect="auto", cmap="magma", norm=TwoSlopeNorm(vmin=0.0, vcenter=0.10, vmax=max(0.20, float(gap.max().max()))), extent=extent)
    best_x = int(np.where(np.isclose(C_GRID, best_c))[0][0])
    best_y = int(np.where(np.isclose(GAMMA_GRID, best_gamma))[0][0])
    for ax in (ax_auc, ax_gap):
        ax.scatter(best_x, best_y, marker="*", s=150, color="white", edgecolor="black", linewidth=0.8, zorder=4)
        ax.set_xticks(np.arange(len(C_GRID)), [display_parameter(v) for v in C_GRID], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(GAMMA_GRID)), [display_parameter(v) for v in GAMMA_GRID])
        ax.set_xlabel("Penalty $C$")
        ax.set_ylabel(r"RBF parameter $\gamma$ (inverse bandwidth)")
    ax_auc.set_title("(a) Mean validation AUROC")
    ax_gap.set_title("(b) Train--validation AUROC gap")
    fig.colorbar(auc_image, ax=ax_auc, fraction=0.046, pad=0.03, format="%.2f")
    fig.colorbar(gap_image, ax=ax_gap, fraction=0.046, pad=0.03, format="%.2f")
    fig.suptitle("RBF grid search and overfitting diagnostic", fontweight="bold")
    stem = FIGURE_DIR / "rbf_kernel_grid_heatmaps"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def mean_roc(
    labels: np.ndarray, scores: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate five fold-specific ROC curves on a common FPR grid."""
    mean_fpr = np.linspace(0.0, 1.0, 201)
    curves = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fpr, tpr, _ = roc_curve(labels[mask], scores[mask])
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        interpolated[-1] = 1.0
        curves.append(interpolated)
    matrix = np.vstack(curves)
    return mean_fpr, matrix.mean(axis=0), matrix.std(axis=0, ddof=1)


def save_roc_figure(predictions: pd.DataFrame, nested: pd.DataFrame) -> None:
    """Compare mean outer-fold ROC curves for linear and RBF models."""
    linear_predictions = pd.read_csv(CSV_DIR / "soft_margin_oof_predictions.csv")
    labels = predictions["true_label"].to_numpy()
    folds = predictions["outer_fold"].to_numpy()
    rbf_fpr, rbf_mean, rbf_std = mean_roc(
        labels, predictions["oof_decision_score_full_precision"].to_numpy(), folds
    )
    linear_fpr, linear_mean, linear_std = mean_roc(
        labels, linear_predictions["oof_decision_score_full_precision"].to_numpy(), folds
    )
    linear_nested = pd.read_csv(CSV_DIR / "soft_margin_nested_cv.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.plot(linear_fpr, linear_mean, color="#1F77B4", linewidth=2.0, label=f"Linear: mean AUC {linear_nested.outer_test_AUROC.mean():.2f}")
    ax.fill_between(linear_fpr, np.clip(linear_mean-linear_std, 0, 1), np.clip(linear_mean+linear_std, 0, 1), color="#1F77B4", alpha=0.12)
    ax.plot(rbf_fpr, rbf_mean, color="#D62728", linewidth=2.0, label=f"RBF: mean AUC {nested.outer_test_AUROC.mean():.2f}")
    ax.fill_between(rbf_fpr, np.clip(rbf_mean-rbf_std, 0, 1), np.clip(rbf_mean+rbf_std, 0, 1), color="#D62728", alpha=0.12)
    ax.plot([0, 1], [0, 1], color="#6B7280", linestyle=":", linewidth=1.2, label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Nested outer-fold ROC: linear versus RBF", xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right")
    stem = FIGURE_DIR / "linear_vs_rbf_nested_roc"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def mark_pareto_frontier(
    frame: pd.DataFrame,
    score_column: str,
    gap_column: str,
) -> np.ndarray:
    """Mark points that maximize score while minimizing overfitting gap."""
    scores = frame[score_column].to_numpy(dtype=float)
    gaps = frame[gap_column].to_numpy(dtype=float)
    frontier = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        weakly_better = (scores >= scores[index]) & (gaps <= gaps[index])
        strictly_better = (scores > scores[index]) | (gaps < gaps[index])
        if np.any(weakly_better & strictly_better):
            frontier[index] = False
    return frontier


def build_pareto_table(rbf_grid: pd.DataFrame) -> pd.DataFrame:
    """Combine the linear and RBF grids and flag non-dominated settings."""
    linear = pd.read_csv(CSV_DIR / "soft_margin_grid_search.csv").copy()
    linear["train_validation_gap"] = (
        linear["mean_train_AUROC"] - linear["mean_validation_AUROC"]
    )
    linear["is_pareto_optimal"] = mark_pareto_frontier(
        linear, "mean_validation_AUROC", "train_validation_gap"
    )
    linear_table = pd.DataFrame(
        {
            "model": "Linear soft margin",
            "C_full_precision": linear["C_full_precision"],
            "gamma_full_precision": np.nan,
            "mean_train_AUROC": linear["mean_train_AUROC"],
            "mean_validation_AUROC": linear["mean_validation_AUROC"],
            "train_validation_gap": linear["train_validation_gap"],
            "is_pareto_optimal": linear["is_pareto_optimal"],
        }
    )

    rbf = rbf_grid.copy()
    rbf["is_pareto_optimal"] = mark_pareto_frontier(
        rbf, "mean_validation_AUROC", "train_validation_gap"
    )
    rbf_table = pd.DataFrame(
        {
            "model": "RBF kernel",
            "C_full_precision": rbf["C_full_precision"],
            "gamma_full_precision": rbf["gamma_full_precision"],
            "mean_train_AUROC": rbf["mean_train_AUROC"],
            "mean_validation_AUROC": rbf["mean_validation_AUROC"],
            "train_validation_gap": rbf["train_validation_gap"],
            "is_pareto_optimal": rbf["is_pareto_optimal"],
        }
    )
    table = pd.concat([linear_table, rbf_table], ignore_index=True)
    table["C_display_2dp"] = table["C_full_precision"].map(display_parameter)
    table["gamma_display_2dp"] = table["gamma_full_precision"].map(
        lambda value: "--" if pd.isna(value) else display_parameter(float(value))
    )
    table["validation_AUROC_display_2dp"] = table[
        "mean_validation_AUROC"
    ].map(lambda value: f"{value:.2f}")
    table["gap_display_2dp"] = table["train_validation_gap"].map(
        lambda value: f"{value:.2f}"
    )
    return table


def save_pareto_figure(table: pd.DataFrame) -> None:
    """Plot validation AUROC versus overfitting gap for both model grids."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_linear, ax_rbf) = plt.subplots(
        1, 2, figsize=(11.2, 4.7), constrained_layout=True
    )

    linear = table[table["model"].eq("Linear soft margin")].copy()
    linear_front = linear[linear["is_pareto_optimal"]]
    linear_scatter = ax_linear.scatter(
        linear["train_validation_gap"],
        linear["mean_validation_AUROC"],
        c=np.log10(linear["C_full_precision"]),
        cmap="viridis",
        s=58,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_linear.scatter(
        linear_front["train_validation_gap"],
        linear_front["mean_validation_AUROC"],
        marker="*",
        s=230,
        color="#D62728",
        edgecolor="black",
        linewidth=0.8,
        label="Pareto-optimal",
        zorder=5,
    )
    ax_linear.annotate(
        "$C=0.01$",
        xy=(
            float(linear_front["train_validation_gap"].iloc[0]),
            float(linear_front["mean_validation_AUROC"].iloc[0]),
        ),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=8.5,
    )
    ax_linear.set_title("(a) Linear soft-margin grid")
    fig.colorbar(
        linear_scatter,
        ax=ax_linear,
        fraction=0.046,
        pad=0.03,
        label=r"$\log_{10}C$",
        format="%.2f",
    )

    rbf = table[table["model"].eq("RBF kernel")].copy()
    rbf_front = rbf[rbf["is_pareto_optimal"]]
    log_c = np.log10(rbf["C_full_precision"].to_numpy())
    sizes = 34.0 + 15.0 * (log_c - log_c.min())
    rbf_scatter = ax_rbf.scatter(
        rbf["train_validation_gap"],
        rbf["mean_validation_AUROC"],
        c=np.log10(rbf["gamma_full_precision"]),
        cmap="plasma",
        s=sizes,
        alpha=0.58,
        edgecolor="white",
        linewidth=0.35,
    )
    ax_rbf.scatter(
        rbf_front["train_validation_gap"],
        rbf_front["mean_validation_AUROC"],
        marker="*",
        s=230,
        color="#D62728",
        edgecolor="black",
        linewidth=0.8,
        label="Pareto-optimal",
        zorder=5,
    )
    ax_rbf.annotate(
        "Pareto plateau:\n4 values of $C$ at $\\gamma=0.00316$",
        xy=(
            float(rbf_front["train_validation_gap"].iloc[0]),
            float(rbf_front["mean_validation_AUROC"].iloc[0]),
        ),
        xytext=(0.52, 0.88),
        textcoords="axes fraction",
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "#4B5563", "linewidth": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
    )
    ax_rbf.text(
        0.98,
        0.04,
        "marker size increases with $C$",
        transform=ax_rbf.transAxes,
        ha="right",
        fontsize=8.2,
        color="#4B5563",
    )
    ax_rbf.set_title(r"(b) RBF $C$--$\gamma$ grid")
    fig.colorbar(
        rbf_scatter,
        ax=ax_rbf,
        fraction=0.046,
        pad=0.03,
        label=r"$\log_{10}\gamma$",
        format="%.2f",
    )

    for ax in (ax_linear, ax_rbf):
        ax.set_xlabel("Train--validation AUROC gap (lower is better)")
        ax.set_ylabel("Mean validation AUROC (higher is better)")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(alpha=0.20)
        ax.legend(loc="lower left", fontsize=8.5)
    fig.suptitle(
        "Pareto frontier: validation performance versus overfitting",
        fontweight="bold",
    )
    stem = FIGURE_DIR / "svm_grid_pareto_frontier"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = frame["Model 1 Target"].eq("UPA").astype(int).to_numpy()
    nested, predictions = nested_search(features, labels)
    search, grid = full_grid_search(features, labels)
    audit, support = audit_model(search, features, labels, frame["PatientID"])
    comparison, metrics = compute_comparison(nested, predictions)
    worst = grid.loc[grid["train_validation_gap"].idxmax()]
    linear_outer_auc = float(comparison.loc[comparison["model"].eq("Linear soft margin"), "mean_outer_AUROC"].iat[0])
    extra_audit = pd.DataFrame(
        [
            ("mean_nested_outer_AUROC", float(nested["outer_test_AUROC"].mean()), f"{nested['outer_test_AUROC'].mean():.2f}"),
            ("std_nested_outer_AUROC", float(nested["outer_test_AUROC"].std(ddof=1)), f"{nested['outer_test_AUROC'].std(ddof=1):.2f}"),
            ("rbf_minus_linear_mean_outer_AUROC", float(nested["outer_test_AUROC"].mean()) - linear_outer_auc, f"{float(nested['outer_test_AUROC'].mean()) - linear_outer_auc:.2e}"),
            ("maximum_grid_train_validation_gap", float(worst["train_validation_gap"]), f"{float(worst['train_validation_gap']):.2f}"),
            ("maximum_gap_C", float(worst["C_full_precision"]), display_parameter(float(worst["C_full_precision"]))),
            ("maximum_gap_gamma", float(worst["gamma_full_precision"]), display_parameter(float(worst["gamma_full_precision"]))),
            ("selected_gamma_at_lower_grid_boundary", float(np.isclose(float(search.best_params_["svc__gamma"]), GAMMA_GRID.min())), "Yes" if np.isclose(float(search.best_params_["svc__gamma"]), GAMMA_GRID.min()) else "No"),
        ],
        columns=audit.columns,
    )
    audit = pd.concat([audit, extra_audit], ignore_index=True)
    pareto = build_pareto_table(grid)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    predictions.insert(1, "PatientID", frame["PatientID"])
    nested.to_csv(CSV_DIR / "rbf_nested_cv.csv", index=False, float_format="%.15g")
    predictions.to_csv(CSV_DIR / "rbf_oof_predictions.csv", index=False, float_format="%.15g")
    grid.to_csv(CSV_DIR / "rbf_grid_search.csv", index=False, float_format="%.15g")
    audit.to_csv(CSV_DIR / "rbf_solution_audit.csv", index=False, float_format="%.15g")
    support.to_csv(CSV_DIR / "rbf_support_vectors.csv", index=False, float_format="%.15g")
    comparison.to_csv(CSV_DIR / "linear_vs_rbf_comparison.csv", index=False, float_format="%.15g")
    pareto.to_csv(CSV_DIR / "svm_grid_pareto_frontier.csv", index=False, float_format="%.15g")
    pd.DataFrame([metrics]).to_csv(CSV_DIR / "rbf_nested_metrics.csv", index=False, float_format="%.15g")

    save_tables(nested, audit, comparison)
    save_grid_figure(
        grid,
        float(search.best_params_["svc__C"]),
        float(search.best_params_["svc__gamma"]),
    )
    save_roc_figure(predictions, nested)
    save_pareto_figure(pareto)

    print(f"Best C: {float(search.best_params_['svc__C']):.12g}")
    print(f"Best gamma: {float(search.best_params_['svc__gamma']):.12g}")
    print(f"Mean outer AUROC: {nested.outer_test_AUROC.mean():.6f}")
    print(comparison.to_string(index=False))
    print(audit.to_string(index=False))
    print("Pareto-optimal grid settings")
    print(
        pareto.loc[
            pareto["is_pareto_optimal"],
            ["model", "C_display_2dp", "gamma_display_2dp", "validation_AUROC_display_2dp", "gap_display_2dp"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

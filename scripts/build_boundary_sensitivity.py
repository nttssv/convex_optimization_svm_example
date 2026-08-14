#!/usr/bin/env python3
"""Audit the lower boundaries of the prespecified linear and RBF grids.

This is an explicitly post-selection sensitivity analysis. It does not replace
the primary nested-CV protocol or use its enlarged grid to revise the headline
performance estimate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import kernel_rbf_svm as rbf
import linear_soft_margin_svm as linear
from linear_hard_margin_svm import FEATURES


ROOT = Path(__file__).resolve().parents[1]
EXTENDED_C = 10.0 ** np.arange(-4.0, 2.01, 0.5)
EXTENDED_GAMMA = 10.0 ** np.arange(-5.0, 1.01, 0.5)


def fit_grid(estimator, parameters, features, labels):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    search = GridSearchCV(
        estimator,
        parameters,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(features, labels)
    return search


def plateau(values: pd.DataFrame, parameter: str) -> tuple[float, float]:
    best = float(values["mean_test_score"].max())
    tied = values.loc[np.isclose(values["mean_test_score"], best, atol=1e-12)]
    return float(tied[parameter].astype(float).min()), float(tied[parameter].astype(float).max())


def latex_scientific(value: float) -> str:
    """Format a positive grid value as compact LaTeX scientific notation."""
    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10.0**exponent)
    if np.isclose(coefficient, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{coefficient:.3g}\!\times\!10^{{{exponent}}}"


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "cgh_pa_dataset.csv")
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = frame["Model 1 Target"].eq("UPA").astype(int).to_numpy()

    linear_search = fit_grid(
        linear.make_pipeline(), {"svc__C": EXTENDED_C}, features, labels
    )
    linear_results = pd.DataFrame(linear_search.cv_results_)
    linear_results = pd.DataFrame(
        {
            "C": linear_results["param_svc__C"].astype(float),
            "mean_train_AUROC": linear_results["mean_train_score"],
            "mean_validation_AUROC": linear_results["mean_test_score"],
            "sd_validation_AUROC_ddof0": linear_results["std_test_score"],
        }
    )
    linear_lo, linear_hi = plateau(
        pd.DataFrame(linear_search.cv_results_), "param_svc__C"
    )

    rbf_search = fit_grid(
        rbf.make_pipeline(),
        {"svc__C": EXTENDED_C, "svc__gamma": EXTENDED_GAMMA},
        features,
        labels,
    )
    rbf_raw = pd.DataFrame(rbf_search.cv_results_)
    rbf_results = pd.DataFrame(
        {
            "C": rbf_raw["param_svc__C"].astype(float),
            "gamma": rbf_raw["param_svc__gamma"].astype(float),
            "mean_train_AUROC": rbf_raw["mean_train_score"],
            "mean_validation_AUROC": rbf_raw["mean_test_score"],
            "sd_validation_AUROC_ddof0": rbf_raw["std_test_score"],
        }
    )
    best_rbf = float(rbf_results["mean_validation_AUROC"].max())
    tied_rbf = rbf_results.loc[
        np.isclose(rbf_results["mean_validation_AUROC"], best_rbf, atol=1e-12)
    ]

    summary = pd.DataFrame(
        [
            {
                "model": "Linear soft margin",
                "prespecified_selection": "C=0.01",
                "extended_grid_selection": f"C={linear_search.best_params_['svc__C']:.6g}",
                "extended_best_validation_AUROC": float(linear_search.best_score_),
                "exact_best_plateau": f"C in [{linear_lo:.6g}, {linear_hi:.6g}]",
                "interpretation": "lower-C plateau; C is not precisely identified",
            },
            {
                "model": "RBF kernel",
                "prespecified_selection": "C=0.01, gamma=0.001",
                "extended_grid_selection": (
                    f"C={rbf_search.best_params_['svc__C']:.6g}, "
                    f"gamma={rbf_search.best_params_['svc__gamma']:.6g}"
                ),
                "extended_best_validation_AUROC": best_rbf,
                "exact_best_plateau": f"{len(tied_rbf)} tied pair(s)",
                "interpretation": "smoother/lower-penalty limit; no unique interior optimum",
            },
        ]
    )

    csv_dir = ROOT / "output" / "csv"
    table_dir = ROOT / "output" / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    linear_results.to_csv(csv_dir / "linear_boundary_sensitivity.csv", index=False, float_format="%.15g")
    rbf_results.to_csv(csv_dir / "rbf_boundary_sensitivity.csv", index=False, float_format="%.15g")
    summary.to_csv(csv_dir / "boundary_sensitivity_summary.csv", index=False, float_format="%.15g")

    linear_lo_tex = latex_scientific(linear_lo)
    linear_hi_tex = latex_scientific(linear_hi)
    rbf_c_tex = latex_scientific(float(rbf_search.best_params_["svc__C"]))
    rbf_gamma_tex = latex_scientific(float(rbf_search.best_params_["svc__gamma"]))

    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Model & Primary grid choice & Extended-grid result & Interpretation \\",
        r"\midrule",
        (
            rf"Linear & $C=0.01$ & best AUROC {linear_search.best_score_:.3f}; "
            rf"$C\in[{linear_lo_tex},{linear_hi_tex}]$ tied & lower-$C$ plateau \\"
        ),
        (
            rf"RBF & $C=0.01,\ \gamma=10^{{-3}}$ & best AUROC {best_rbf:.3f}; "
            rf"{len(tied_rbf)}-setting tie; smallest $C={rbf_c_tex},\ \gamma={rbf_gamma_tex}$ "
            rf"& smoother-limit solution \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (table_dir / "boundary_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

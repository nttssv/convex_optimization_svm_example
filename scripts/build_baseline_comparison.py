#!/usr/bin/env python3
"""Compare the SVMs with PAC-only and linear logistic baselines over 20 seeds."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import linear_soft_margin_svm as linear
from linear_hard_margin_svm import FEATURES


ROOT = Path(__file__).resolve().parents[1]
LOGISTIC_C = 10.0 ** np.arange(-4.0, 2.01, 0.5)


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("log_transform", FunctionTransformer(linear.log_selected_columns, validate=False)),
            ("median_imputer", SimpleImputer(strategy="median")),
            ("standard_scaler", StandardScaler()),
            ("model", LogisticRegression(solver="lbfgs", max_iter=10_000)),
        ]
    )


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "cgh_pa_dataset.csv")
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = frame["Model 1 Target"].eq("UPA").astype(int).to_numpy()
    rows = []

    for seed in range(20):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pac_fold_auc = []
        logistic_fold_auc = []
        for fold, (train, test) in enumerate(outer.split(features, labels), start=1):
            # PAC is a prespecified monotone score, so it has no fitted parameter.
            pac_fold_auc.append(roc_auc_score(labels[test], features.iloc[test]["PAC"]))
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
            search = GridSearchCV(
                logistic_pipeline(),
                {"model__C": LOGISTIC_C},
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
            )
            search.fit(features.iloc[train], labels[train])
            logistic_fold_auc.append(
                roc_auc_score(labels[test], search.predict_proba(features.iloc[test])[:, 1])
            )
        rows.extend(
            [
                {"seed": seed, "model": "PAC only", "mean_outer_AUROC": np.mean(pac_fold_auc)},
                {"seed": seed, "model": "L2 logistic", "mean_outer_AUROC": np.mean(logistic_fold_auc)},
            ]
        )

    results = pd.DataFrame(rows)
    svm = pd.read_csv(ROOT / "output" / "csv" / "linear_rbf_seed_robustness.csv")
    svm_rows = pd.concat(
        [
            svm[["seed", "linear_mean_outer_AUROC"]]
            .rename(columns={"linear_mean_outer_AUROC": "mean_outer_AUROC"})
            .assign(model="Linear SVM"),
            svm[["seed", "rbf_mean_outer_AUROC"]]
            .rename(columns={"rbf_mean_outer_AUROC": "mean_outer_AUROC"})
            .assign(model="RBF SVM"),
        ],
        ignore_index=True,
    )
    results = pd.concat([results, svm_rows], ignore_index=True)
    order = ["PAC only", "L2 logistic", "Linear SVM", "RBF SVM"]
    summary = (
        results.groupby("model")["mean_outer_AUROC"]
        .agg(["mean", "std", "min", "max"])
        .reindex(order)
        .reset_index()
        .rename(columns={"std": "seed_sd_ddof1"})
    )

    csv_dir = ROOT / "output" / "csv"
    table_dir = ROOT / "output" / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_dir / "baseline_seed_comparison.csv", index=False, float_format="%.15g")
    summary.to_csv(csv_dir / "baseline_comparison_summary.csv", index=False, float_format="%.15g")

    body = [
        f"{row.model} & {row.mean:.3f} & {row.seed_sd_ddof1:.3f} & {row.min:.3f}--{row.max:.3f} \\\\"
        for row in summary.itertuples(index=False)
    ]
    table = "\n".join(
        [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Model & Mean AUROC & Seed SD & Seed range \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    (table_dir / "baseline_comparison_table.tex").write_text(table, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

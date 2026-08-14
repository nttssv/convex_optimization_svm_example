#!/usr/bin/env python3
"""Repeat nested CV across split seeds and plot descriptive stability."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import kernel_rbf_svm as rbf
import linear_soft_margin_svm as linear
from linear_hard_margin_svm import FEATURES


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    frame = pd.read_csv(ROOT / "data" / "cgh_pa_dataset.csv")
    features = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    labels = frame["Model 1 Target"].eq("UPA").astype(int).to_numpy()
    rows = []
    for seed in range(20):
        linear.RANDOM_STATE = seed
        rbf.RANDOM_STATE = seed
        linear_folds, _, _ = linear.nested_grid_search(features, labels)
        rbf_folds, _ = rbf.nested_search(features, labels)
        linear_auc = float(linear_folds["outer_test_AUROC"].mean())
        rbf_auc = float(rbf_folds["outer_test_AUROC"].mean())
        rows.append({"seed": seed, "linear_mean_outer_AUROC": linear_auc, "rbf_mean_outer_AUROC": rbf_auc, "linear_minus_rbf": linear_auc - rbf_auc})
        print(f"seed {seed:02d}: linear={linear_auc:.6f}, rbf={rbf_auc:.6f}")

    results = pd.DataFrame(rows)
    csv_dir = ROOT / "output" / "csv"
    fig_dir = ROOT / "output" / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_dir / "linear_rbf_seed_robustness.csv", index=False, float_format="%.15g")

    differences = results["linear_minus_rbf"].to_numpy()
    mean_difference = float(differences.mean())
    lower, upper = np.quantile(differences, [0.025, 0.975])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), gridspec_kw={"width_ratios": [1.25, 0.75]})
    axes[0].plot(results["seed"], results["linear_mean_outer_AUROC"], "o-", label="Linear", color="#1f77b4", lw=1.6, ms=4)
    axes[0].plot(results["seed"], results["rbf_mean_outer_AUROC"], "s-", label="RBF", color="#e4572e", lw=1.6, ms=4)
    axes[0].set_xlabel("Cross-validation split seed")
    axes[0].set_ylabel("Mean outer-fold AUROC")
    axes[0].set_xticks(range(0, 20, 2))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    colors = np.where(differences >= 0, "#1f77b4", "#e4572e")
    jitter = np.linspace(-0.07, 0.07, len(differences))
    axes[1].scatter(jitter, differences, c=colors, s=28, alpha=0.9)
    axes[1].errorbar(0, mean_difference, yerr=[[mean_difference - lower], [upper - mean_difference]], fmt="o", color="black", capsize=5, lw=1.5)
    axes[1].axhline(0, color="0.45", ls="--", lw=1)
    axes[1].set_xlim(-0.2, 0.2)
    axes[1].set_xticks([])
    axes[1].set_ylabel("AUROC difference: linear - RBF")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(0.03, 0.97, f"Mean = {mean_difference:+.4f}\nDescriptive 95% interval\n[{lower:+.3f}, {upper:+.3f}]", transform=axes[1].transAxes, va="top", fontsize=8)

    fig.suptitle("Seed robustness of mean nested outer-fold AUROC", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stat_seed_robustness.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"mean difference={mean_difference:+.6f}; descriptive interval=[{lower:+.6f}, {upper:+.6f}]")


if __name__ == "__main__":
    main()

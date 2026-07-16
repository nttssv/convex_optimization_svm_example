#!/usr/bin/env python3
"""Build and verify the five-point primal--dual hard-margin SVM example."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import _pubstyle; _pubstyle.apply()
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "figures"
OUTPUT_STEM = OUTPUT_DIR / "svm_primal_dual_toy_example"

NAMES = np.array(["A", "B", "C", "D", "E"])
X = np.array(
    [
        [-2.0, 0.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [2.0, 0.0],
        [2.0, 2.0],
    ]
)
Y = np.array([-1.0, -1.0, 1.0, 1.0, 1.0])

# Analytic primal and dual solutions derived in the report.
W = np.array([1.0, 0.0])
B = 0.0
ALPHA = np.array([0.0, 0.5, 0.5, 0.0, 0.0])


def verify_example() -> tuple[np.ndarray, float, float]:
    """Return margins and objectives after checking the KKT identities."""
    margins = Y * (X @ W + B)
    primal_objective = 0.5 * float(W @ W)
    reconstructed_w = (ALPHA * Y) @ X
    dual_objective = float(ALPHA.sum() - 0.5 * reconstructed_w @ reconstructed_w)

    np.testing.assert_allclose(margins, [2.0, 1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(reconstructed_w, W)
    np.testing.assert_allclose(ALPHA @ Y, 0.0)
    np.testing.assert_allclose(ALPHA * (margins - 1.0), 0.0)
    np.testing.assert_allclose(primal_objective, 0.5)
    np.testing.assert_allclose(dual_objective, primal_objective)
    return margins, primal_objective, dual_objective


def build_figure() -> None:
    """Plot the decision boundary, margins, classes, and support vectors."""
    margins, primal_objective, dual_objective = verify_example()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.axvspan(-1.0, 1.0, color="#6B7280", alpha=0.08, zorder=0)
    ax.axvline(0.0, color="#111827", linewidth=2.2, label=r"Decision boundary $x_1=0.00$")
    ax.axvline(-1.0, color="#6B7280", linestyle="--", linewidth=1.6)
    ax.axvline(1.0, color="#6B7280", linestyle="--", linewidth=1.6)

    negative = Y == -1
    positive = Y == 1
    ax.scatter(
        X[negative, 0],
        X[negative, 1],
        s=78,
        marker="o",
        color="#1F77B4",
        edgecolor="white",
        linewidth=0.8,
        label=r"Class $y=-1$",
        zorder=3,
    )
    ax.scatter(
        X[positive, 0],
        X[positive, 1],
        s=82,
        marker="D",
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.8,
        label=r"Class $y=+1$",
        zorder=3,
    )

    support = ALPHA > 0
    ax.scatter(
        X[support, 0],
        X[support, 1],
        s=230,
        facecolors="none",
        edgecolors="#111827",
        linewidths=2.0,
        zorder=4,
    )

    offsets = {
        "A": (-0.13, 0.16),
        "B": (-0.18, 0.16),
        "C": (0.10, 0.16),
        "D": (0.10, 0.16),
        "E": (0.10, 0.16),
    }
    for name, (x1, x2), margin in zip(NAMES, X, margins, strict=True):
        dx, dy = offsets[str(name)]
        ax.text(x1 + dx, x2 + dy, f"{name}  ($m={margin:.2f}$)", fontsize=9.5)

    ax.text(-1.0, -0.43, r"$x_1=-1.00$", ha="center", color="#4B5563")
    ax.text(1.0, -0.43, r"$x_1=1.00$", ha="center", color="#4B5563")
    ax.text(0.0, 2.38, r"$x_1=0.00$", ha="center", color="#111827")

    support_handle = Line2D(
        [0],
        [0],
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgecolor="#111827",
        markeredgewidth=1.8,
        linestyle="None",
        label="Support vector (B, C)",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(support_handle)
    labels.append(support_handle.get_label())

    ax.set(
        xlabel=r"Feature $x_1$",
        ylabel=r"Feature $x_2$",
        title="Five-point hard-margin SVM: boundary, margins, and support vectors",
        xlim=(-2.6, 2.65),
        ylim=(-0.55, 2.55),
        xticks=np.arange(-2, 3, 1),
        yticks=np.arange(0, 3, 1),
    )
    ax.grid(alpha=0.18)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
        fontsize=8.8,
    )
    ax.text(
        0.02,
        0.98,
        rf"$p^*={primal_objective:.2f}=d^*={dual_objective:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#D1D5DB"},
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()

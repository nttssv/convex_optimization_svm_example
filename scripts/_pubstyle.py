"""Shared publication figure style for the CGH SVM report.

Unifies typography, sizing, and grid appearance across all report figures so
they share a consistent identity and match the Latin Modern body text
(Computer Modern math via mathtext 'cm'). Deliberately does NOT set the color
cycle: the model scripts assign UPA/BPA and other series colors explicitly, and
overriding the cycle here would silently change those. Purely cosmetic; no
figure geometry, data, or numerical output is affected.

Call apply() once, immediately after importing matplotlib.pyplot.
"""
import matplotlib as mpl


def apply():
    mpl.rcParams.update({
        # typography — serif body with Computer Modern math to match the report
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        # frame & grid — lighter, consistent
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # legend — quiet box
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.4,
        # output fidelity (kept from the scripts' own settings)
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

#!/usr/bin/env python3
"""Generate the descriptive-statistics LaTeX tables used by the report."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA = Path("data/cgh_pa_dataset.csv")
OUT = Path("output/tables")
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
LOG_FEATURES = {"PAC", "PRA", "18-OHF", "18-oxoF"}


def fmt_signed(value: float) -> str:
    return f"{value:+.2f}"


def build_overall(df: pd.DataFrame) -> str:
    rows = []
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce").dropna().to_numpy()
        q1, q3 = np.quantile(values, [0.25, 0.75])
        log_skew = fmt_signed(stats.skew(np.log(values + 1e-6), bias=False)) if feature in LOG_FEATURES else "--"
        rows.append(
            f"{feature} & {values.mean():.2f} & {np.median(values):.2f} & "
            f"{q1:.2f}--{q3:.2f} & {values.min():.2f}--{values.max():.2f} & "
            f"{fmt_signed(stats.skew(values, bias=False))} & {log_skew} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Predictor & Mean & Median & IQR & Range & Skew & Skew (log) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def build_by_class(df: pd.DataFrame) -> str:
    rows = []
    for feature in FEATURES:
        upa = pd.to_numeric(df.loc[df["Model 1 Target"] == "UPA", feature], errors="coerce").dropna().to_numpy()
        bpa = pd.to_numeric(df.loc[df["Model 1 Target"] == "BPA", feature], errors="coerce").dropna().to_numpy()
        pooled_sd = np.sqrt(((len(upa) - 1) * upa.var(ddof=1) + (len(bpa) - 1) * bpa.var(ddof=1)) / (len(upa) + len(bpa) - 2))
        effect = (upa.mean() - bpa.mean()) / pooled_sd
        test = stats.ttest_ind(upa, bpa, equal_var=False)
        p_display = r"$<0.001$" if test.pvalue < 0.001 else f"{test.pvalue:.3f}"
        rows.append(
            f"{feature} & {upa.mean():.1f} ({upa.std(ddof=1):.1f}) & "
            f"{bpa.mean():.1f} ({bpa.std(ddof=1):.1f}) & {effect:+.2f} & "
            f"{test.statistic:+.2f} & {p_display} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Predictor & UPA mean (SD) & BPA mean (SD) & Cohen's $d$ & Welch $t$ & $p$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "descriptive_statistics_table.tex").write_text(build_overall(df), encoding="utf-8")
    (OUT / "descriptive_statistics_by_class_table.tex").write_text(build_by_class(df), encoding="utf-8")
    print("Generated both descriptive-statistics table fragments.")


if __name__ == "__main__":
    main()

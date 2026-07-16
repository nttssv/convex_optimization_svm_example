#!/usr/bin/env python3
"""Generate a synthetic CGH-like primary aldosteronism (PA) dataset.

The real patient-level CGH dataset is not available in this project. The course
rubric explicitly permits synthetic data ("Use real (if available) or synthetic
data"). This script builds a 114-patient cohort whose marginal statistics match
the aggregate values reported in the CGH midterm report (mean age 48, mean PAC
25.1 ng/dL, mean potassium 3.59 mmol/L, mean PRA 0.71 ng/mL/h, mean tumor size
14 mm, mean 18-oxoF 88.9 ng/dL, mean 18-OHF 1319 ng/dL, mean systolic BP 141
mmHg, mean antihypertensive burden 2.39 drugs).

A clinically plausible signal separates unilateral PA (UPA) from bilateral PA
(BPA): UPA cases tend to have higher aldosterone (PAC), more suppressed renin
(PRA), lower potassium, larger adrenal tumors, and higher hybrid steroids
(18-OHF, 18-oxoF). Column names match exactly what
scripts/svm_diagnostics_from_data.py expects.

Usage:
    python3 scripts/make_synthetic_dataset.py
Writes:
    data/cgh_pa_dataset.csv
    data/cgh_pa_dataset.xlsx   (if openpyxl is available)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260710
N = 114


def lognormal_with_mean(rng, target_mean, sigma, class_shift, size):
    """Draw a lognormal variable whose (unshifted) mean is ``target_mean``.

    ``class_shift`` is an additive offset applied in log space so that the two
    subtypes differ; it is (approximately) mean-centred across the cohort so the
    overall marginal mean stays close to ``target_mean``.
    """
    mu = np.log(target_mean) - 0.5 * sigma ** 2
    return np.exp(rng.normal(mu, sigma, size) + class_shift)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Roughly balanced subtype split (unilateral vs bilateral PA).
    y = rng.binomial(1, 0.5, N)  # 1 = UPA (surgically curable), 0 = BPA
    # Centred class indicator so additive shifts leave the cohort mean ~unchanged.
    s = y - y.mean()

    # --- Demographic / clinical -------------------------------------------------
    age = np.clip(rng.normal(48.0, 11.0, N) - 3.0 * s, 21, 72)
    sex = rng.binomial(1, 0.55 + 0.05 * s, N)  # 1 = male, mild UPA predominance
    sbp = np.clip(rng.normal(141.0, 17.0, N) + 6.0 * s, 100, 200)
    dbp = np.clip(rng.normal(88.0, 11.0, N) + 4.0 * s, 60, 130)
    ddd = np.clip(np.round(rng.normal(2.39, 1.1, N) + 0.5 * s), 0, 6)

    # --- Biochemistry -----------------------------------------------------------
    potassium = np.clip(rng.normal(3.59, 0.48, N) - 0.30 * s, 2.3, 5.2)
    pac = lognormal_with_mean(rng, 25.1, 0.55, 0.55 * s, N)        # higher in UPA
    pra = lognormal_with_mean(rng, 0.71, 0.75, -0.60 * s, N)       # suppressed in UPA

    # --- Imaging ----------------------------------------------------------------
    tumor = np.clip(rng.normal(14.0, 6.0, N) + 5.0 * s, 0, 45)

    # --- Steroidomics (hybrid steroids, right-skewed) --------------------------
    ohf = lognormal_with_mean(rng, 1319.0, 0.65, 0.70 * s, N)      # 18-OHF
    oxof = lognormal_with_mean(rng, 88.9, 0.70, 0.75 * s, N)       # 18-oxoF

    df = pd.DataFrame(
        {
            "PatientID": [f"CGH-{i+1:03d}" for i in range(N)],
            "Age": np.round(age, 0).astype(int),
            "Sex": sex.astype(int),
            "Systolic BP": np.round(sbp, 0).astype(int),
            "Diastolic BP": np.round(dbp, 0).astype(int),
            "DDD": ddd.astype(int),
            "Potassium": np.round(potassium, 2),
            "PAC": np.round(pac, 1),
            "PRA": np.round(pra, 2),
            "Tumor size": np.round(tumor, 1),
            "18-OHF": np.round(ohf, 0).astype(int),
            "18-oxoF": np.round(oxof, 1),
            "Model 1 Target": np.where(y == 1, "UPA", "BPA"),
        }
    )

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cgh_pa_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")

    try:
        xlsx_path = out_dir / "cgh_pa_dataset.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"Wrote {xlsx_path}")
    except Exception as exc:  # openpyxl missing, etc.
        print(f"Skipped Excel export: {exc}")

    # Quick marginal check vs. reported cohort statistics.
    print("\nMarginal means (synthetic vs reported):")
    reported = {
        "Age": 48.0, "Systolic BP": 141.0, "DDD": 2.39, "Potassium": 3.59,
        "PAC": 25.1, "PRA": 0.71, "Tumor size": 14.0, "18-OHF": 1319.0, "18-oxoF": 88.9,
    }
    for col, ref in reported.items():
        print(f"  {col:<14} {df[col].mean():>10.2f}   (reported {ref})")
    print(f"\nUPA n={int((df['Model 1 Target']=='UPA').sum())}, "
          f"BPA n={int((df['Model 1 Target']=='BPA').sum())}")


if __name__ == "__main__":
    main()

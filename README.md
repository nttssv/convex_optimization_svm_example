# SVM project draft bundle

This folder is a self-contained snapshot of the current report and its
reproducible artifacts.

## Main files

- `cgh_hard_margin_project_report.tex`: report source.
- `output/pdf/cgh_hard_margin_project_report.pdf`: final rendered report.
- `data/`: CSV and spreadsheet versions of the synthetic project dataset.
- `scripts/`: hard-margin, linear soft-margin, and RBF-kernel analyses.
- `output/csv/`: full-precision numerical results and two-decimal display
  columns used by the report.
- `output/csv/svm_grid_pareto_frontier.csv`: linear and RBF grid settings,
  dominance status, and Pareto-optimal settings.
- `output/tables/`: generated LaTeX table fragments.
- `output/figures/`: report figures in PDF/PNG format.
- `output/figures/svm_grid_pareto_frontier.*`: reproducible Pareto chart for
  validation AUROC versus the train--validation AUROC gap.
- `output/xlsx/`: ten-observation teaching workbook.

## Reproduce the analyses

Run these commands from this `draft` directory, with `scripts/` on
`PYTHONPATH` (the soft-margin and RBF scripts import the sibling module
`linear_hard_margin_svm`, so they fail with `ModuleNotFoundError` if it is
not importable):

```bash
export PYTHONPATH="$PWD/scripts"
python3 scripts/linear_hard_margin_svm.py
python3 scripts/build_full_data_infeasibility_certificate.py
python3 scripts/build_svm_primal_dual_toy_example.py
python3 scripts/build_hard_margin_teaching_workbook.py
python3 scripts/linear_soft_margin_svm.py
python3 scripts/kernel_rbf_svm.py
```

The scripts read the dataset via paths relative to this `draft` directory,
so they must be launched from here (not from inside `scripts/`). On systems
where `GridSearchCV(n_jobs=-1)` cannot create POSIX semaphores (some
sandboxed or container environments raise `PermissionError`), prepend
`JOBLIB_MULTIPROCESSING=0` to force the sequential backend; results are
numerically identical.

## Rebuild the PDF

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=output/pdf cgh_hard_margin_project_report.tex
```

The primary model conclusion is that the linear soft-margin SVM is preferred:
the RBF kernel did not improve mean nested outer-fold AUROC and showed severe
overfitting for large values of gamma.  The Pareto analysis is a tuning
diagnostic and does not replace the untouched nested outer-fold comparison.

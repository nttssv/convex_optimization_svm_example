#!/usr/bin/env python3
"""Generate SVM optimization diagnostics from a patient-level dataset.

Example:
    python3 scripts/svm_diagnostics_from_data.py \
        --data data/cgh_pa_dataset.csv \
        --label "Model 1 Target" \
        --features "PAC,PRA,Potassium,Tumor size,18-OHF,18-oxoF" \
        --log-features "PAC,PRA,18-OHF,18-oxoF"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute squared-hinge SVM loss, Hessian eigenvalues, L-smooth constant, and learning-rate curves."
    )
    parser.add_argument("--data", required=True, help="Path to CSV or Excel patient-level dataset.")
    parser.add_argument("--label", required=True, help="Binary target column, e.g. UPA/BPA or 1/0.")
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated feature columns. If omitted, all numeric columns except label are used.",
    )
    parser.add_argument(
        "--log-features",
        default="",
        help="Comma-separated positive continuous columns to log-transform before standardization.",
    )
    parser.add_argument("--lambda-reg", type=float, default=1e-2, help="L2 regularization parameter.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Gradient descent iterations.")
    parser.add_argument("--out-dir", default="output/figures", help="Output directory for figures and summaries.")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data type: {suffix}. Use CSV or Excel.")


def split_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def encode_binary_label(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(dtype=float)
        unique = sorted(pd.unique(values[~pd.isna(values)]))
        if set(unique).issubset({0.0, 1.0}):
            return values.astype(int)
        if len(unique) == 2:
            return (values == unique[-1]).astype(int)
        raise ValueError(f"Numeric label column must have exactly two values; got {unique}.")

    normalized = series.astype(str).str.strip().str.lower()
    positive = {"1", "upa", "unilateral", "unilateral pa", "positive", "yes", "true"}
    negative = {"0", "bpa", "bilateral", "bilateral pa", "negative", "no", "false"}

    y = np.full(len(series), -1, dtype=int)
    y[normalized.isin(positive)] = 1
    y[normalized.isin(negative)] = 0
    if np.any(y < 0):
        bad = sorted(set(normalized[y < 0]))
        raise ValueError(f"Could not map label values to binary classes: {bad}")
    return y


def prepare_xy(df: pd.DataFrame, label_col: str, feature_cols: list[str], log_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    if not feature_cols:
        feature_cols = [
            col for col in df.columns
            if col != label_col and pd.api.types.is_numeric_dtype(df[col])
        ]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")

    y = encode_binary_label(df[label_col])
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    for col in log_cols:
        if col not in X_df.columns:
            raise ValueError(f"Log-transform column not found in features: {col}")
        min_value = X_df[col].min(skipna=True)
        offset = 1e-6 if min_value > 0 else abs(min_value) + 1e-6
        X_df[col] = np.log(X_df[col] + offset)

    X_df = X_df.fillna(X_df.median(numeric_only=True))
    means = X_df.mean(axis=0)
    stds = X_df.std(axis=0, ddof=0).replace(0, 1.0)
    X = ((X_df - means) / stds).to_numpy(dtype=float)
    return X, y, feature_cols


def class_weights(y01: np.ndarray) -> np.ndarray:
    n = len(y01)
    counts = np.bincount(y01.astype(int), minlength=2)
    weights = np.ones(n)
    for cls in (0, 1):
        if counts[cls] > 0:
            weights[y01 == cls] = n / (2.0 * counts[cls])
    return weights


def svm_loss_and_grad(X: np.ndarray, t: np.ndarray, w: np.ndarray, b: float, lam: float, weights: np.ndarray) -> tuple[float, np.ndarray, float]:
    n = X.shape[0]
    scores = X @ w + b
    margin = 1.0 - t * scores
    active = margin > 0
    hinge = np.maximum(0.0, margin)
    loss = 0.5 * lam * np.dot(w, w) + np.sum(weights * hinge**2) / n

    grad_w = lam * w
    grad_b = 0.0
    if np.any(active):
        weighted = weights[active] * margin[active]
        grad_w -= (2.0 / n) * (X[active].T @ (weighted * t[active]))
        grad_b -= (2.0 / n) * np.sum(weighted * t[active])
    return loss, grad_w, grad_b


def hessian_for_active_set(X: np.ndarray, t: np.ndarray, w: np.ndarray, b: float, lam: float, weights: np.ndarray, all_samples: bool = False) -> np.ndarray:
    n, p = X.shape
    U = t[:, None] * np.column_stack([X, np.ones(n)])
    theta = np.r_[w, b]
    if all_samples:
        active = np.ones(n, dtype=bool)
    else:
        active = 1.0 - U @ theta > 0.0

    H = np.zeros((p + 1, p + 1))
    H[:p, :p] = lam * np.eye(p)
    H += (2.0 / n) * (U[active].T @ (weights[active, None] * U[active]))
    return H


def fit_gd(X: np.ndarray, y01: np.ndarray, lam: float, lr: float, max_iter: int, weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, bool]:
    t = 2 * y01.astype(float) - 1.0
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    stable = True
    for _ in range(max_iter):
        loss, grad_w, grad_b = svm_loss_and_grad(X, t, w, b, lam, weights)
        losses.append(loss)
        if not np.isfinite(loss) or loss > 1e8:
            stable = False
            break
        w -= lr * grad_w
        b -= lr * grad_b
    return b, w, np.asarray(losses), stable


def objective_along_solution(X: np.ndarray, y01: np.ndarray, b: float, w: np.ndarray, lam: float, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = 2 * y01.astype(float) - 1.0
    scales = np.linspace(-0.5, 1.75, 160)
    losses = []
    for scale in scales:
        loss, _, _ = svm_loss_and_grad(X, t, scale * w, scale * b, lam, weights)
        losses.append(loss)
    return scales, np.asarray(losses)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(Path(args.data))
    feature_cols = split_columns(args.features)
    log_cols = split_columns(args.log_features)
    X, y01, used_features = prepare_xy(df, args.label, feature_cols, log_cols)
    weights = class_weights(y01)
    t = 2 * y01.astype(float) - 1.0

    H_global = hessian_for_active_set(
        X, t, np.zeros(X.shape[1]), 0.0, args.lambda_reg, weights, all_samples=True
    )
    global_eigs = np.linalg.eigvalsh(H_global)
    L_global = float(global_eigs[-1])
    safe_lr = 1.0 / L_global

    lr_grid = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.90]) * safe_lr
    histories = []
    summary_rows = []
    best = None
    for lr in lr_grid:
        b, w, losses, stable = fit_gd(X, y01, args.lambda_reg, float(lr), args.max_iter, weights)
        final_loss = float(losses[-1]) if len(losses) else np.inf
        histories.append((lr, losses, stable))
        summary_rows.append({"learning_rate": lr, "multiple_of_1_over_L": lr / safe_lr, "final_loss": final_loss, "stable": stable})
        if stable and (best is None or final_loss < best[0]):
            best = (final_loss, lr, b, w)

    if best is None:
        raise RuntimeError("All learning rates diverged. Check preprocessing and labels.")

    final_loss, best_lr, best_b, best_w = best
    H_active = hessian_for_active_set(X, t, best_w, best_b, args.lambda_reg, weights)
    active_eigs = np.linalg.eigvalsh(H_active)
    mu_active = float(max(active_eigs[0], 0.0))
    L_active = float(active_eigs[-1])
    alpha_quad = float(2.0 / (L_active + mu_active)) if mu_active > 1e-12 else np.nan

    # Figure 1: loss along fitted SVM direction.
    scales, loss_slice = objective_along_solution(X, y01, best_b, best_w, args.lambda_reg, weights)
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(scales, loss_slice, linewidth=2)
    plt.axvline(1.0, color="black", linestyle="--", linewidth=1, label="fitted SVM direction")
    plt.xlabel("Scale c applied to fitted parameters (cw, cb)")
    plt.ylabel("Squared-hinge SVM objective")
    plt.title("SVM loss along fitted parameter direction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "svm_loss_slice.png", dpi=220)
    plt.close()

    # Figure 2: Hessian eigenvalues.
    plt.figure(figsize=(7.0, 4.4))
    x_axis = np.arange(len(active_eigs))
    plt.semilogy(x_axis, np.maximum(active_eigs, 1e-12), marker="o", linewidth=1.5)
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Active-set Hessian eigenvalues")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "hessian_eigenvalues.png", dpi=220)
    plt.close()

    # Figure 3: learning-rate comparison.
    plt.figure(figsize=(7.2, 4.6))
    for lr, losses, stable in histories:
        label = f"{lr / safe_lr:.2g}/L"
        if not stable:
            label += " diverged"
        plt.plot(losses, linewidth=1.6, label=label)
    plt.xlabel("Gradient descent iteration")
    plt.ylabel("Training objective")
    plt.title("Learning-rate comparison")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate_comparison.png", dpi=220)
    plt.close()

    pd.DataFrame(summary_rows).to_csv(out_dir / "learning_rate_summary.csv", index=False)
    np.savetxt(out_dir / "hessian_eigenvalues.csv", active_eigs, delimiter=",")

    with open(out_dir / "svm_diagnostics_summary.txt", "w", encoding="utf-8") as handle:
        handle.write("SVM optimization diagnostics\n")
        handle.write("============================\n")
        handle.write(f"Data file: {args.data}\n")
        handle.write(f"Label column: {args.label}\n")
        handle.write(f"Feature columns: {', '.join(used_features)}\n")
        handle.write(f"Number of samples: {X.shape[0]}\n")
        handle.write(f"Number of features: {X.shape[1]}\n")
        handle.write(f"lambda: {args.lambda_reg:.6g}\n")
        handle.write(f"L-smooth upper bound: {L_global:.6g}\n")
        handle.write(f"Safe learning rate 1/L: {safe_lr:.6g}\n")
        handle.write(f"Best empirical learning rate: {best_lr:.6g}\n")
        handle.write(f"Best empirical final loss: {final_loss:.6g}\n")
        handle.write(f"Active Hessian min eigenvalue: {active_eigs[0]:.6g}\n")
        handle.write(f"Active Hessian max eigenvalue: {active_eigs[-1]:.6g}\n")
        handle.write(f"Local quadratic alpha*: {alpha_quad:.6g}\n")
        handle.write(f"Convex check: min eigenvalue >= -1e-8 is {active_eigs[0] >= -1e-8}\n")


if __name__ == "__main__":
    main()

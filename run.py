# run.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
import json

def train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_test = int(test_size * n)
    n_val  = int(val_size * n)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_standardizer(X_train):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0.0] = 1.0
    return mu, sigma

def apply_standardizer(X, mu, sigma):
    return (X - mu) / sigma

def parse_grid(grid_str):
    return [int(x) for x in grid_str.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to housing.csv")
    ap.add_argument("--mode", required=True, choices=["fixed", "grid"])
    ap.add_argument("--k", type=int, help="k for mode=fixed")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data/processed")
    args = ap.parse_args()

    # Basic arg checks
    if args.mode == "fixed" and args.k is None:
        ap.error("--k is required when --mode fixed")
    if args.mode == "grid" and not args.k_grid:
        ap.error("--k-grid is required when --mode grid (e.g., 1,3,5,7,9)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load and prepare features/target (numeric-only for simplicity)
    df = pd.read_csv(args.csv).dropna()
    target_col = "median_house_value"
    y = df[target_col].to_numpy(dtype=float)
    X = df.drop(columns=[target_col]).select_dtypes(include="number").to_numpy(dtype=float)

    # ---- One reproducible split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=args.val_size, test_size=args.test_size, random_state=args.seed
    )

    # ---- Standardize with train stats only
    mu, sigma = fit_standardizer(X_train)
    X_train = apply_standardizer(X_train, mu, sigma)
    X_val   = apply_standardizer(X_val,   mu, sigma)
    X_test  = apply_standardizer(X_test,  mu, sigma)

    # ---- Persist standardized splits for all runners (scratch & sklearn)
    splits_path = outdir / "splits_standardized.npz"
    np.savez(
        splits_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        mu=mu, sigma=sigma  # handy for future framework parity
    )

    cmd = [
        sys.executable, "-m", "src.run_knn",
        "--splits", str(splits_path),
        "--mode", args.mode,
        "--seed", str(args.seed)
    ]
    if args.mode == "fixed":
        cmd += ["--k", str(args.k)]
    else:
        cmd += ["--k-grid", args.k_grid]

    print("\n[run.py] Running my KNN model...")
    ret = subprocess.call(cmd)
    if ret != 0:
        sys.exit(ret)

    # ---- Call sklearn baseline with the SAME splits/args
    cmd_fw = [
        sys.executable, "-m", "src.framework_knn",
        "--splits", str(splits_path),
        "--mode", args.mode,
        "--seed", str(args.seed)
    ]
    if args.mode == "fixed":
        cmd_fw += ["--k", str(args.k)]
    else:
        cmd_fw += ["--k-grid", args.k_grid]

    print("\n[run.py] Running sklearn KNN baseline...")
    ret2 = subprocess.call(cmd_fw)
    if ret2 != 0:
        sys.exit(ret2)

if __name__ == "__main__":
    main()

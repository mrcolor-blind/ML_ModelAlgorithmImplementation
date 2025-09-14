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
    ap.add_argument("--csv", required=False, help="Path to housing.csv (or prediction inputs in predict mode)")
    ap.add_argument("--mode", required=True, choices=["grid", "predict"])
    ap.add_argument("--k", type=int, help="k for mode=predict")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data/processed")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits_path = outdir / "splits_standardized.npz"

    if args.mode == "grid":
        if not args.csv:
            ap.error("--csv (training dataset) is required for --mode grid")
        if not args.k_grid:
            ap.error("--k-grid is required for --mode grid (e.g., 1,3,5,7,9)")

        # ---- Load and prepare features/target (numeric-only)
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

        # ---- Persist standardized splits and scaler
        np.savez(
            splits_path,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            mu=mu, sigma=sigma
        )

        # ---- Run scratch
        cmd = [
            sys.executable, "-m", "src.run_knn",
            "--splits", str(splits_path),
            "--mode", "grid",
            "--k-grid", args.k_grid,
            "--seed", str(args.seed)
        ]
        print("\n[run.py] Running your KNN (grid)...")
        ret = subprocess.call(cmd); 
        if ret != 0: sys.exit(ret)

        # ---- Run sklearn
        cmd_fw = [
            sys.executable, "-m", "src.framework_knn",
            "--splits", str(splits_path),
            "--mode", "grid",
            "--k-grid", args.k_grid,
            "--seed", str(args.seed)
        ]
        print("\n[run.py] Running sklearn KNN (grid)...")
        ret2 = subprocess.call(cmd_fw)
        if ret2 != 0: sys.exit(ret2)

    elif args.mode == "predict":
        if args.k is None:
            ap.error("--k is required for --mode predict")
        if not args.csv:
            ap.error("--csv (feature-only inputs) is required for --mode predict")
        if not splits_path.exists():
            ap.error(f"{splits_path} not found. Run grid mode first to create splits and scaler.")

        # ---- Scratch predictions
        cmd = [
            sys.executable, "-m", "src.run_knn",
            "--splits", str(splits_path),
            "--mode", "predict",
            "--k", str(args.k),
            "--predict-csv", args.csv
        ]
        print("\n[run.py] Predict with your KNN...")
        ret = subprocess.call(cmd); 
        if ret != 0: sys.exit(ret)

        # ---- Sklearn predictions
        cmd_fw = [
            sys.executable, "-m", "src.framework_knn",
            "--splits", str(splits_path),
            "--mode", "predict",
            "--k", str(args.k),
            "--predict-csv", args.csv
        ]
        print("\n[run.py] Predict with sklearn KNN...")
        ret2 = subprocess.call(cmd_fw)
        if ret2 != 0: sys.exit(ret2)

if __name__ == "__main__":
    main()

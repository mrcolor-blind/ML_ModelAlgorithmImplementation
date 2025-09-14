# src/run_knn.py
import argparse
import numpy as np
import pandas as pd
from .knn import KNNRegressor

# ---------- tiny utilities (still here for back-compat) ----------
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

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1.0 - ss_res / ss_tot)

def parse_grid(grid_str):
    return [int(x) for x in grid_str.split(",") if x.strip()]

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    # Preferred path: receive standardized splits produced by run.py
    ap.add_argument("--splits", type=str, help="Path to standardized splits .npz from run.py")
    ap.add_argument("--mode", required=True, choices=["fixed", "grid"])
    ap.add_argument("--k", type=int, help="k for mode=fixed")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--seed", type=int, default=42)

    # Backward-compat path (optional): run directly from CSV like your original script
    ap.add_argument("--csv", type=str, help="Path to housing.csv (fallback mode)")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    args = ap.parse_args()

    if args.mode == "fixed" and args.k is None:
        ap.error("--k is required when --mode fixed")
    if args.mode == "grid" and not args.k_grid:
        ap.error("--k-grid is required when --mode grid")

    # ----- Load data (preferred: splits from run.py)
    if args.splits:
        data = np.load(args.splits, allow_pickle=False)
        X_train = data["X_train"]; y_train = data["y_train"]
        X_val   = data["X_val"];   y_val   = data["y_val"]
        X_test  = data["X_test"];  y_test  = data["y_test"]
    else:
        if not args.csv:
            ap.error("Provide --splits (preferred) or --csv (fallback).")
        df = pd.read_csv(args.csv).dropna()
        target_col = "median_house_value"
        y = df[target_col].to_numpy(dtype=float)
        X = df.drop(columns=[target_col]).select_dtypes(include="number").to_numpy(dtype=float)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
            X, y, val_size=args.val_size, test_size=args.test_size, random_state=args.seed
        )
        mu, sigma = fit_standardizer(X_train)
        X_train = apply_standardizer(X_train, mu, sigma)
        X_val   = apply_standardizer(X_val,   mu, sigma)
        X_test  = apply_standardizer(X_test,  mu, sigma)

    # ----- Train/validate/test
    if args.mode == "fixed":
        k = int(args.k)
        model = KNNRegressor(k=k).fit(X_train, y_train)
        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)
        print(f"k={k:>2} | val MAE={mae(y_val, val_pred):,.2f}")
        print("\n[My Model] Fixed-k results")
        print(f"Test RMSE: {rmse(y_test, test_pred):,.2f}")
        print(f"Test MAE : {mae(y_test, test_pred):,.2f}")
        print(f"Test R^2 : {r2(y_test, test_pred):,.4f}")

    else:  # grid
        k_grid = parse_grid(args.k_grid)
        best = {"k": None, "mae": float("inf")}
        for k in k_grid:
            model = KNNRegressor(k=k).fit(X_train, y_train)
            val_pred = model.predict(X_val)
            score = mae(y_val, val_pred)
            print(f"k={k:>2} | val MAE={score:,.2f}")
            if score < best["mae"]:
                best = {"k": k, "mae": score}

        final = KNNRegressor(k=best["k"]).fit(X_train, y_train)
        test_pred = final.predict(X_test)
        print(f"\nBest k: {best['k']}")
        print("[My Model] Grid-search results (best-k on test)")
        print(f"Test RMSE: {rmse(y_test, test_pred):,.2f}")
        print(f"Test MAE : {mae(y_test, test_pred):,.2f}")
        print(f"Test R^2 : {r2(y_test, test_pred):,.4f}")

if __name__ == "__main__":
    main()

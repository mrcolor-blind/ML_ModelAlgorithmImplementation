# src/framework_knn.py
import argparse
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1.0 - ss_res / ss_tot)

def parse_grid(grid_str):
    return [int(x) for x in grid_str.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="Path to standardized splits .npz from run.py")
    ap.add_argument("--mode", required=True, choices=["fixed", "grid"])
    ap.add_argument("--k", type=int, help="k for mode=fixed")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "fixed" and args.k is None:
        ap.error("--k is required when --mode fixed")
    if args.mode == "grid" and not args.k_grid:
        ap.error("--k-grid is required when --mode grid")

    data = np.load(args.splits, allow_pickle=False)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    X_test  = data["X_test"];  y_test  = data["y_test"]

    if args.mode == "fixed":
        k = int(args.k)
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)
        print(f"k={k:>2} | val MAE={mae(y_val, val_pred):,.2f}")
        print("\n[Sklearn] Fixed-k results")
        print(f"Test RMSE: {rmse(y_test, test_pred):,.2f}")
        print(f"Test MAE : {mae(y_test, test_pred):,.2f}")
        print(f"Test R^2 : {r2(y_test, test_pred):,.4f}")

    else:  # grid
        k_grid = parse_grid(args.k_grid)
        best = {"k": None, "mae": float("inf")}
        for k in k_grid:
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            score = mae(y_val, val_pred)
            print(f"k={k:>2} | val MAE={score:,.2f}")
            if score < best["mae"]:
                best = {"k": k, "mae": score}

        final = KNeighborsRegressor(n_neighbors=best["k"])
        final.fit(X_train, y_train)
        test_pred = final.predict(X_test)
        print(f"\nBest k: {best['k']}")
        print("[Sklearn] Grid-search results (best-k on test)")
        print(f"Test RMSE: {rmse(y_test, test_pred):,.2f}")
        print(f"Test MAE : {mae(y_test, test_pred):,.2f}")
        print(f"Test R^2 : {r2(y_test, test_pred):,.4f}")

if __name__ == "__main__":
    main()

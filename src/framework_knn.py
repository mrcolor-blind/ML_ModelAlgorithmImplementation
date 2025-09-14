# src/framework_knn.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import validation_curve, KFold

# ---------- metrics ----------
def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1.0 - ss_res / ss_tot)

def parse_grid(grid_str):
    return [int(x) for x in grid_str.split(",") if x.strip()]

def _project_root():
    # src/ -> project root
    return Path(__file__).resolve().parents[1]

def _results_dir():
    d = Path(__file__).resolve().parents[1] / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="Path to standardized splits .npz from run.py")
    ap.add_argument("--mode", required=True, choices=["grid", "predict"])
    ap.add_argument("--k", type=int, help="k for mode=predict")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--predict-csv", type=str, help="Feature-only CSV for predictions (predict mode)")
    args = ap.parse_args()

    data = np.load(args.splits, allow_pickle=False)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    X_test  = data["X_test"];  y_test  = data["y_test"]
    mu      = data["mu"];      sigma   = data["sigma"]

    if args.mode == "predict":
        if args.k is None:
            ap.error("--k is required for --mode predict")
        if not args.predict_csv:
            ap.error("--predict-csv is required for --mode predict")

        # Fit on train, standardize incoming features with saved mu/sigma
        model = KNeighborsRegressor(n_neighbors=int(args.k)).fit(X_train, y_train)
        import pandas as pd
        df_pred = pd.read_csv(args.predict_csv)
        X_new = df_pred.select_dtypes(include="number").to_numpy(dtype=float)
        X_new_std = (X_new - mu) / sigma
        y_hat = model.predict(X_new_std)

        out = _results_dir() / "predictions_sklearn.csv"
        pd.DataFrame({"prediction": y_hat}).to_csv(out, index=False)
        print(f"[Sklearn] Saved predictions → {out}")
        return
        
    else:  # mode == "grid"
        k_grid = parse_grid(args.k_grid)

        # We'll still print val MAE per k on the provided (train, val) holdout
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

        # ---- Validation curve (MSE vs k) using CV on train+val ----
        # ADD THESE THREE LINES ⬇⬇⬇
        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
        # ⬆⬆⬆

        param_range = np.array(k_grid, dtype=int)
        tr_scores, va_scores = validation_curve(
            estimator=KNeighborsRegressor(),
            X=X_tv, y=y_tv,
            param_name="n_neighbors",
            param_range=param_range,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        train_mean = -np.mean(tr_scores, axis=1)
        train_std  =  np.std(tr_scores,  axis=1)
        val_mean   = -np.mean(va_scores, axis=1)
        val_std    =  np.std(va_scores,  axis=1)

        # Plot & save
        plt.figure()
        plt.plot(param_range, train_mean, label="Entrenamiento")
        plt.fill_between(param_range, train_mean-train_std, train_mean+train_std, alpha=0.2)

        plt.plot(param_range, val_mean, label="Validación")
        plt.fill_between(param_range, val_mean-val_std, val_mean+val_std, alpha=0.2)

        plt.xlabel("Número de vecinos (k)")
        plt.ylabel("MSE")
        plt.title("Curva de validación - KNN (MSE vs k)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = _results_dir() / "validationCurve_withFramework_gridk.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Sklearn] Saved validation curve → {out_path}")

if __name__ == "__main__":
    main()

# src/framework_knn.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve, validation_curve, KFold
from sklearn.dummy import DummyRegressor

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
    d = _project_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="Path to standardized splits .npz from run.py")
    ap.add_argument("--mode", required=True, choices=["fixed", "grid"])
    ap.add_argument("--k", type=int, help="k for mode=fixed")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5, help="CV folds for curves (default: 5)")
    args = ap.parse_args()

    if args.mode == "fixed" and args.k is None:
        ap.error("--k is required when --mode fixed")
    if args.mode == "grid" and not args.k_grid:
        ap.error("--k-grid is required when --mode grid")

    data = np.load(args.splits, allow_pickle=False)
    X_train = data["X_train"]; y_train = data["y_train"]
    X_val   = data["X_val"];   y_val   = data["y_val"]
    X_test  = data["X_test"];  y_test  = data["y_test"]

    # We'll use train+val for any CV-based curves; keep test untouched for final report
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    # A consistent CV splitter (stratification not applicable in regression)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

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

        # ---- Learning curve (bias/variance view) on train+val only ----
        # scoring is neg MSE -> we flip sign to MSE+
        # scoring is neg MSE -> we flip sign to MSE+
        train_sizes = np.linspace(0.1, 1.0, 10)
        sizes_abs, tr_scores, va_scores = learning_curve(
            estimator=KNeighborsRegressor(n_neighbors=k),
            X=X_tv, y=y_tv,
            train_sizes=train_sizes,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            shuffle=True,
            random_state=args.seed
        )


        train_mean = -np.mean(tr_scores, axis=1)
        train_std  =  np.std(tr_scores,  axis=1)
        val_mean   = -np.mean(va_scores, axis=1)
        val_std    =  np.std(va_scores,  axis=1)

        # Baselines (DummyRegressor) for reference
        dummy_mean = DummyRegressor(strategy="mean")
        dm_scores = []
        for tr_idx, te_idx in cv.split(X_tv):
            dummy_mean.fit(X_tv[tr_idx], y_tv[tr_idx])
            yhat = dummy_mean.predict(X_tv[te_idx])
            dm_scores.append(np.mean((y_tv[te_idx] - yhat) ** 2))
        baseline_mean = float(np.mean(dm_scores))

        dummy_median = DummyRegressor(strategy="median")
        dmed_scores = []
        for tr_idx, te_idx in cv.split(X_tv):
            dummy_median.fit(X_tv[tr_idx], y_tv[tr_idx])
            yhat = dummy_median.predict(X_tv[te_idx])
            dmed_scores.append(np.mean((y_tv[te_idx] - yhat) ** 2))
        baseline_median = float(np.mean(dmed_scores))

        # Plot & save
        plt.figure()
        plt.plot(sizes_abs, train_mean, label="Entrenamiento (KNN)")
        plt.fill_between(sizes_abs, train_mean-train_std, train_mean+train_std, alpha=0.2)

        plt.plot(sizes_abs, val_mean, label="Validación (KNN)")
        plt.fill_between(sizes_abs, val_mean-val_std, val_mean+val_std, alpha=0.2)


        plt.axhline(y=baseline_mean, linestyle="--", label="Dummy (mean)")
        plt.axhline(y=baseline_median, linestyle="--", label="Dummy (median)")

        plt.xlabel("Tamaño del conjunto de entrenamiento (observaciones)")
        plt.ylabel("MSE")
        plt.title(f"Curva de aprendizaje - KNN (k={k})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = _results_dir() / "learningCurve_withFramework_fixedk.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[Sklearn] Saved learning curve → {out_path}")

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
        param_range = np.array(k_grid, dtype=int)
        # validation_curve returns arrays [len(param_range), n_splits]
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

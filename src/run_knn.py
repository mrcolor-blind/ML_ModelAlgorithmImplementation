# src/run_knn.py
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from .knn import KNNRegressor

# ---------- tiny utilities ----------
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

def results_dir():
    d = Path(__file__).resolve().parents[1] / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---------- simple CV helpers ----------
def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def cross_val_mse(X, y, model_k, cv=5, seed=42):
    folds = kfold_indices(len(X), k=cv, seed=seed)
    mse_tr, mse_va = [], []
    for i in range(cv):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(cv) if j != i])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[val_idx], y[val_idx]
        model = KNNRegressor(k=model_k).fit(Xtr, ytr)
        yhat_tr = model.predict(Xtr)
        yhat_va = model.predict(Xva)
        mse_tr.append(np.mean((ytr - yhat_tr) ** 2))
        mse_va.append(np.mean((yva - yhat_va) ** 2))
    return np.mean(mse_tr), np.std(mse_tr), np.mean(mse_va), np.std(mse_va)

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=str, help="Path to standardized splits .npz from run.py")
    ap.add_argument("--mode", required=True, choices=["fixed", "grid"])
    ap.add_argument("--k", type=int, help="k for mode=fixed")
    ap.add_argument("--k-grid", type=str, help="comma-separated ks for mode=grid (e.g., 1,3,5,7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv", type=int, default=5, help="CV folds for curves")
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

        # ---- Learning curve on train+val ----
        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        train_sizes = np.linspace(0.1, 1.0, 8)
        tr_mean, tr_std, va_mean, va_std, sizes = [], [], [], [], []
        for frac in train_sizes:
            n_sub = int(len(X_tv) * frac)
            X_sub, y_sub = X_tv[:n_sub], y_tv[:n_sub]
            m_tr, s_tr, m_va, s_va = cross_val_mse(X_sub, y_sub, model_k=k, cv=args.cv, seed=args.seed)
            sizes.append(n_sub)
            tr_mean.append(m_tr); tr_std.append(s_tr)
            va_mean.append(m_va); va_std.append(s_va)

        plt.figure()
        plt.plot(sizes, tr_mean, label="Entrenamiento (MyKNN)")
        plt.fill_between(sizes, np.array(tr_mean)-np.array(tr_std), np.array(tr_mean)+np.array(tr_std), alpha=0.2)
        plt.plot(sizes, va_mean, label="Validación (MyKNN)")
        plt.fill_between(sizes, np.array(va_mean)-np.array(va_std), np.array(va_mean)+np.array(va_std), alpha=0.2)
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("MSE")
        plt.title(f"Curva de aprendizaje - MyKNN (k={k})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = results_dir() / "learningCurve_myKNN_fixedk.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"[My Model] Saved learning curve → {out_path}")

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

        # ---- Validation curve on train+val ----
        X_tv = np.vstack([X_train, X_val])
        y_tv = np.concatenate([y_train, y_val])
        tr_mean, tr_std, va_mean, va_std = [], [], [], []
        for k in k_grid:
            m_tr, s_tr, m_va, s_va = cross_val_mse(X_tv, y_tv, model_k=k, cv=args.cv, seed=args.seed)
            tr_mean.append(m_tr); tr_std.append(s_tr)
            va_mean.append(m_va); va_std.append(s_va)

        plt.figure()
        plt.plot(k_grid, tr_mean, label="Entrenamiento (MyKNN)")
        plt.fill_between(k_grid, np.array(tr_mean)-np.array(tr_std), np.array(tr_mean)+np.array(tr_std), alpha=0.2)
        plt.plot(k_grid, va_mean, label="Validación (MyKNN)")
        plt.fill_between(k_grid, np.array(va_mean)-np.array(va_std), np.array(va_mean)+np.array(va_std), alpha=0.2)
        plt.xlabel("Número de vecinos (k)")
        plt.ylabel("MSE")
        plt.title("Curva de validación - MyKNN")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = results_dir() / "validationCurve_myKNN_grid.png"
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        print(f"[My Model] Saved validation curve → {out_path}")

if __name__ == "__main__":
    main()

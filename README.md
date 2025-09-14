# ML Model Algorithm Implementation

## KNN Regression Model – From Scratch vs Scikit-learn

This repository contains:

1. **KNN Regression from scratch** – implemented fully in Python, without using any ML frameworks or libraries.  
2. **KNN Regression with Scikit-learn** – using `KNeighborsRegressor` on the **exact same splits and standardization** to allow a fair comparison.  

Both implementations share the same data preparation, splits, and scaling.  
The repo also includes **automation scripts** (`run.py`) to orchestrate the entire pipeline:  
- Automatically splits data into train/validation/test sets.  
- Applies feature standardization based on the training set.  
- Runs **MyKNN** (from scratch), outputs validation results, and saves a validation curve plot.  
- Runs **Scikit-learn KNN**, outputs validation results, and saves its validation curve plot.  
- Supports a **predict mode** to generate predictions from a CSV file using both implementations, saving results side-by-side.

---

## Dependencies

- **Python**: 3.9+ recommended  
- **Required libraries**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

Install with:
```
pip install -r requirements.txt
```

##  Input Data

1. **`housing.csv`**  
   - California Housing dataset with the following features:  
     `longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value`  
   - Used for training, validation, and testing.  
   - The target variable is **`median_house_value`**.  

2. **`predictionInputs.csv`**  
   - Contains rows with the **same feature structure as the training data, excluding the target column** (`median_house_value`).  
   - Example:

   ```
   longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income
   -122.23,37.88,41,880,129,322,126,8.3252
   -118.25,34.05,30,2000,300,900,350,5.1234
   ```

## How to Run

### 1. Train & Validate (Grid Search)

Runs the pipeline end-to-end: split → standardize → run **MyKNN** → plot → run **Scikit-learn KNN** → plot.

```
`python run.py --mode grid --csv data/housing.csv --k-grid 1,3,5,7,9,15,25`
```

### 2. Predict with Both Models

Uses the saved training split + scaler to predict from a **feature-only CSV**.

```
`python run.py --mode predict --csv data/predictionInputs.csv --k 7`
```

---

## Outputs

1. **Console logs** with validation MAE for each `k`, best selected `k`, and final test results (RMSE, MAE, R²).  
2. **Plots** (saved in `/results`):  
   - `validationCurve_myKNN_grid.png`  
   - `validationCurve_withFramework_gridk.png`  
3. **Predictions** (saved in `/results`):  
   - `predictions_myKNN.csv`  
   - `predictions_sklearn.csv`  

## Input Command Arguments

The main script `run.py` accepts the following arguments:

- `--mode`  
  - **Required**.  
  - Options:  
    - `grid` → runs the full training + validation pipeline (splitting, standardizing, MyKNN + Sklearn comparison, plots).  
    - `predict` → runs prediction on new inputs using both MyKNN and Sklearn, outputs CSVs with predictions.  

- `--csv`  
  - **Required**.  
  - In `grid` mode → path to the **training dataset** (e.g., `data/housing.csv`).  
  - In `predict` mode → path to the **feature-only dataset** (e.g., `data/predictionInputs.csv`).  

- `--k-grid`  
  - **Required only in grid mode**.  
  - Comma-separated list of neighbor values to test (e.g., `1,3,5,7,9,15,25`).  
  - Used for validation curve and model selection.  

- `--k`  
  - **Required only in predict mode**.  
  - Integer value for the number of neighbors to use when generating predictions.  

- `--val-size`  
  - Optional.  
  - Fraction of the dataset reserved for the validation split (default: `0.15`).  

- `--test-size`  
  - Optional.  
  - Fraction of the dataset reserved for the test split (default: `0.15`).  

- `--seed`  
  - Optional.  
  - Random seed for reproducibility of splits and cross-validation (default: `42`).  

- `--outdir`  
  - Optional.  
  - Directory where processed splits will be saved (default: `data/processed`).  

---


---

This repository demonstrates not only how to implement KNN regression **from scratch**, but also how to validate its correctness against a trusted ML library, automate the workflow, and extend it to real prediction tasks.

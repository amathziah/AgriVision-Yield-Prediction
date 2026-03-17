"""
==============================================================
  Satellite-Based Precision Agriculture
  SVR — RBF Kernel Hyperparameter Tuning (GridSearchCV)
==============================================================
Extends src/svr_model.py by adding:
  • GridSearchCV over C × gamma × epsilon
  • 3-fold cross-validation
  • Default RBF vs Tuned RBF comparison
  • Updated results table + comparison plot

Run:
    python src/svr_tuning.py

Dependencies: pandas, numpy, scikit-learn, matplotlib
"""

import sys
import textwrap
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Import shared steps from svr_model ────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from svr_model import (
    load_data,
    encode_categoricals,
    split_features_target,
    split_and_subsample,
    DATA_PATH,
    RESULTS_DIR,
    TARGET_COL,
    CAT_COLS,
    DROP_COLS,
    TEST_SIZE,
    RANDOM_STATE,
    SUBSAMPLE_N,
    SVR_CONFIGS,
)

SEP = "=" * 64

# ──────────────────────────────────────────────────────────
# TUNING CONFIGURATION
# ──────────────────────────────────────────────────────────

# Search-space for GridSearchCV
#   Keys are prefixed with "svr__" because the SVR sits inside
#   a Pipeline whose step is named "svr".
PARAM_GRID: dict[str, list] = {
    "svr__C"      : [1, 10, 100],
    "svr__gamma"  : ["scale", 0.01, 0.1],
    "svr__epsilon": [0.1, 0.5, 1.0],
}

CV_FOLDS          = 3           # k-fold cross-validation
SCORING_METRIC    = "neg_root_mean_squared_error"   # maximised → lower RMSE
N_JOBS            = -1          # use all available CPU cores
TUNING_SUBSAMPLE  = 5_000       # smaller sample for grid search (3×27 fits)
PLOT_N            = 500         # scatter-plot sample size


# ──────────────────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────────────────

def _sec(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ──────────────────────────────────────────────────────────
# STEP 1 — DEFAULT RBF BASELINE
# ──────────────────────────────────────────────────────────

def train_default_rbf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Train the default RBF SVR (C=10, gamma='scale', epsilon=0.1)
    as a baseline to compare against the tuned version.
    """
    _sec("STEP 1 — DEFAULT RBF BASELINE")

    default_params = SVR_CONFIGS["RBF"]
    print(f"  Default params: {default_params}\n")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(**default_params)),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  ✅  Default RBF trained in {elapsed:.1f}s")
    print(f"       MAE  = {mae:.4f} t/ha")
    print(f"       RMSE = {rmse:.4f} t/ha")

    return {"label": "RBF (default)", "pipeline": pipeline,
            "y_pred": y_pred, "mae": mae, "rmse": rmse,
            "params": default_params}


# ──────────────────────────────────────────────────────────
# STEP 2 — GRIDSEARCHCV TUNING
# ──────────────────────────────────────────────────────────

def run_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int,
    n_jobs: int,
    tuning_subsample: int,
    random_state: int,
) -> GridSearchCV:
    """
    Exhaustive grid search over param_grid using k-fold CV.

    ╔══════════════════════════════════════════════════════╗
    ║  What GridSearchCV does mathematically               ║
    ╠══════════════════════════════════════════════════════╣
    ║  For each candidate θ in the Cartesian product of   ║
    ║  PARAM_GRID (27 combinations here):                 ║
    ║                                                     ║
    ║    1. Split training set into k folds               ║
    ║    2. For fold i ∈ {1…k}:                          ║
    ║         • Train model on k−1 folds                  ║
    ║         • Evaluate on held-out fold i               ║
    ║         • Record score s(i, θ)                      ║
    ║    3. CV score(θ) = mean over k folds               ║
    ║                                                     ║
    ║    Best θ* = argmax_θ CV score(θ)                   ║
    ║                                                     ║
    ║  Total fits = |grid| × k = 27 × 3 = 81 fits        ║
    ╚══════════════════════════════════════════════════════╝

    Uses a subsample for speed (SVR is O(n²) — tuning on
    the full set would require 81 × O(n²) fits).

    Returns
    -------
    Fitted GridSearchCV object.
    """
    _sec("STEP 2 — GRIDSEARCHCV  (27 combos × 3 folds = 81 fits)")

    # Subsample for grid search
    n = min(tuning_subsample, len(X_train))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_train), size=n, replace=False)
    X_gs = X_train.iloc[idx]
    y_gs = y_train.iloc[idx]

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    print(f"  Search space   : {param_grid}")
    print(f"  Combinations   : {total_combos}")
    print(f"  CV folds       : {cv}")
    print(f"  Total fits     : {total_combos * cv}")
    print(f"  Scoring metric : {SCORING_METRIC}")
    print(f"  Tuning sample  : {n} rows\n")

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=3000)),
    ])

    grid_search = GridSearchCV(
        estimator  = base_pipeline,
        param_grid = param_grid,
        cv         = cv,
        scoring    = SCORING_METRIC,
        n_jobs     = n_jobs,
        verbose    = 2,
        refit      = True,          # retrain best estimator on full X_gs
        return_train_score = True,
    )

    t0 = time.perf_counter()
    grid_search.fit(X_gs, y_gs)
    elapsed = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────
    best_params = {
        k.replace("svr__", ""): v
        for k, v in grid_search.best_params_.items()
    }
    best_cv_rmse = -grid_search.best_score_    # stored as negative

    print(f"\n  ⏱️   Grid search completed in {elapsed:.1f}s")
    print(f"\n  🏆  Best parameters found:")
    for k, v in best_params.items():
        print(f"       {k:<10} = {v}")
    print(f"\n  📊  Best CV RMSE = {best_cv_rmse:.4f} t/ha")

    return grid_search


# ──────────────────────────────────────────────────────────
# STEP 3 — RETRAIN ON FULL TRAIN SET WITH BEST PARAMS
# ──────────────────────────────────────────────────────────

def retrain_tuned_rbf(
    grid_search: GridSearchCV,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Refit the best pipeline from GridSearchCV on the full
    training set (not just the tuning subsample) to get the
    most accurate test-set evaluation.

    WHY RETRAIN ON FULL DATA?
    ──────────────────────────
    GridSearchCV.refit=True refits the winner on the grid-search
    subsample.  For maximum model quality we re-fit it again on
    ALL available training data — the search only selected θ*,
    the final fit uses every training row we have.
    """
    _sec("STEP 3 — RETRAIN TUNED MODEL ON FULL TRAIN SET")

    best_params_raw = grid_search.best_params_   # still has "svr__" prefix
    best_params = {k.replace("svr__", ""): v for k, v in best_params_raw.items()}

    print(f"  Retraining RBF SVR with: {best_params}")

    tuned_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=5000, **best_params)),
    ])

    t0 = time.perf_counter()
    tuned_pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = tuned_pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  ✅  Tuned RBF retrained in {elapsed:.1f}s")
    print(f"       MAE  = {mae:.4f} t/ha")
    print(f"       RMSE = {rmse:.4f} t/ha")

    return {"label": "RBF (tuned)", "pipeline": tuned_pipeline,
            "y_pred": y_pred, "mae": mae, "rmse": rmse,
            "params": best_params}


# ──────────────────────────────────────────────────────────
# STEP 4 — CV RESULTS HEATMAP
# ──────────────────────────────────────────────────────────

def plot_cv_heatmap(grid_search: GridSearchCV) -> None:
    """
    Visualise the mean CV RMSE across the C × gamma grid,
    one subplot per epsilon value.  Helps see the smoothness
    of the hyperparameter landscape.
    """
    _sec("STEP 4 — CV SCORE HEATMAP")

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results["mean_test_rmse"] = -cv_results["mean_test_score"]

    epsilons = sorted(cv_results["param_svr__epsilon"].unique())
    C_vals   = sorted(cv_results["param_svr__C"].unique())
    gammas   = list(cv_results["param_svr__gamma"].unique())

    n_eps = len(epsilons)
    fig, axes = plt.subplots(1, n_eps, figsize=(6 * n_eps, 5), sharey=True)
    if n_eps == 1:
        axes = [axes]

    fig.suptitle("GridSearchCV — Mean CV RMSE (t/ha) across C × gamma",
                 fontsize=13, fontweight="bold")

    vmin = cv_results["mean_test_rmse"].min()
    vmax = cv_results["mean_test_rmse"].max()

    for ax, eps in zip(axes, epsilons):
        subset = cv_results[cv_results["param_svr__epsilon"] == eps]
        pivot = subset.pivot_table(
            index="param_svr__C",
            columns="param_svr__gamma",
            values="mean_test_rmse",
            aggfunc="mean",
        )

        im = ax.imshow(pivot.values, aspect="auto",
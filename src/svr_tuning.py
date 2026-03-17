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
import joblib

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
                       cmap="RdYlGn_r", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(g) for g in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(c) for c in pivot.index], fontsize=9)
        ax.set_xlabel("gamma", fontsize=10)
        ax.set_ylabel("C", fontsize=10)
        ax.set_title(f"epsilon = {eps}", fontsize=11, fontweight="bold")

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=axes[-1], label="Mean CV RMSE (t/ha)")
    plt.tight_layout()

    path = RESULTS_DIR / "svr_gridsearch_heatmap.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📊  Heatmap saved → '{path}'")
    plt.close()


# ──────────────────────────────────────────────────────────
# STEP 5 — COMPARISON PLOT
# ──────────────────────────────────────────────────────────

def plot_comparison(
    default_result: dict,
    tuned_result: dict,
    y_test: pd.Series,
    plot_n: int,
) -> None:
    """
    Side-by-side Predicted vs Actual scatter for Default vs Tuned RBF.
    Also adds a bar chart comparing MAE and RMSE.
    """
    _sec("STEP 5 — COMPARISON PLOT: DEFAULT vs TUNED RBF")

    rng    = np.random.RandomState(42)
    idx    = rng.choice(len(y_test), size=min(plot_n, len(y_test)), replace=False)
    y_samp = np.array(y_test)[idx]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Default RBF  vs  Tuned RBF — Crop Yield Prediction",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    color_default = "#4C72B0"
    color_tuned   = "#55A868"

    for col_idx, result in enumerate([default_result, tuned_result]):
        color  = color_default if col_idx == 0 else color_tuned
        y_pred = result["y_pred"][idx]

        # -- Predicted vs Actual --
        ax = fig.add_subplot(gs[0, col_idx])
        ax.scatter(y_samp, y_pred, alpha=0.4, s=18,
                   color=color, edgecolors="none")
        lims = [
            min(y_samp.min(), y_pred.min()) - 0.5,
            max(y_samp.max(), y_pred.max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", lw=1.2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual yield (t/ha)", fontsize=10)
        ax.set_ylabel("Predicted yield (t/ha)", fontsize=10)
        ax.set_title(
            f"{result['label']}\nMAE={result['mae']:.3f}  "
            f"RMSE={result['rmse']:.3f}",
            fontsize=11, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.35)

        # -- Residuals --
        ax_res = fig.add_subplot(gs[1, col_idx])
        residuals = y_samp - y_pred
        ax_res.scatter(y_pred, residuals, alpha=0.4, s=18,
                       color=color, edgecolors="none")
        ax_res.axhline(0, color="k", lw=1.2, linestyle="--")
        ax_res.set_xlabel("Fitted values", fontsize=10)
        ax_res.set_ylabel("Residuals (t/ha)", fontsize=10)
        ax_res.set_title(f"Residuals — {result['label']}",
                         fontsize=11, fontweight="bold")
        ax_res.grid(True, linestyle="--", alpha=0.35)

    # -- Metric bar chart --
    ax_bar = fig.add_subplot(gs[:, 2])
    labels   = [default_result["label"], tuned_result["label"]]
    mae_vals = [default_result["mae"],   tuned_result["mae"]]
    rmse_vals= [default_result["rmse"],  tuned_result["rmse"]]

    x = np.arange(len(labels))
    width = 0.35
    b1 = ax_bar.bar(x - width/2, mae_vals,  width, label="MAE",  color="#4C72B0", alpha=0.85)
    b2 = ax_bar.bar(x + width/2, rmse_vals, width, label="RMSE", color="#C44E52", alpha=0.85)

    ax_bar.bar_label(b1, fmt="%.3f", fontsize=9, padding=3)
    ax_bar.bar_label(b2, fmt="%.3f", fontsize=9, padding=3)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_ylabel("Error (t/ha)", fontsize=10)
    ax_bar.set_title("MAE & RMSE Comparison", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=10)
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bar.set_ylim(0, max(rmse_vals) * 1.3)

    plt.tight_layout()
    path = RESULTS_DIR / "svr_default_vs_tuned.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📊  Comparison plot saved → '{path}'")
    plt.close()


# ──────────────────────────────────────────────────────────
# STEP 6 — UPDATED RESULTS TABLE
# ──────────────────────────────────────────────────────────

def print_results_table(
    default_result: dict,
    tuned_result: dict,
    all_kernel_results: dict,
) -> None:
    """
    Print the full updated comparison — all original kernels
    plus Default RBF and Tuned RBF, ranked by RMSE.
    """
    _sec("STEP 6 — UPDATED FULL RESULTS TABLE")

    rows = []

    # Original kernel baselines (from svr_model.py run)
    for name, res in all_kernel_results.items():
        rows.append({
            "Model"     : f"SVR — {name}",
            "MAE (t/ha)": res["mae"],
            "RMSE(t/ha)": res["rmse"],
            "Note"      : "baseline",
        })

    # Tuned RBF (override the default RBF row)
    rows.append({
        "Model"     : "SVR — RBF (tuned)",
        "MAE (t/ha)": tuned_result["mae"],
        "RMSE(t/ha)": tuned_result["rmse"],
        "Note"      : f"C={tuned_result['params'].get('C')}  "
                      f"γ={tuned_result['params'].get('gamma')}  "
                      f"ε={tuned_result['params'].get('epsilon')}",
    })

    df = pd.DataFrame(rows).sort_values("RMSE(t/ha)")
    df["Rank"] = range(1, len(df) + 1)
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    df["Rank"] = df["Rank"].map(lambda r: f"{medals.get(r, '')}  #{r}")

    print(df.to_string(index=False))

    best = df.iloc[0]
    mae_improvement  = default_result["mae"]  - tuned_result["mae"]
    rmse_improvement = default_result["rmse"] - tuned_result["rmse"]

    print(f"\n  🏆  Best overall model: {best['Model']}")
    print(f"\n  📈  Improvement over default RBF:")
    print(f"       MAE  reduced by {mae_improvement:+.4f} t/ha  "
          f"({100*mae_improvement/default_result['mae']:.1f}%)")
    print(f"       RMSE reduced by {rmse_improvement:+.4f} t/ha  "
          f"({100*rmse_improvement/default_result['rmse']:.1f}%)")


# ──────────────────────────────────────────────────────────
# STEP 7 — THEORY EXPLANATION
# ──────────────────────────────────────────────────────────

def print_theory() -> None:
    _sec("STEP 7 — THEORY: GridSearchCV + Bias-Variance Tradeoff")

    text = textwrap.dedent("""
  ┌────────────────────────────────────────────────────────────┐
  │ 1. WHAT GridSearchCV DOES MATHEMATICALLY                   │
  └────────────────────────────────────────────────────────────┘
  Goal: find θ* = argmin_{θ ∈ Θ} E[L(y, f_θ(x))]

  Since E[L] is unknowable, we estimate it using k-fold CV:

    CV_score(θ) = (1/k) × Σᵢ L(y_val_i, f_θ(x_val_i))

  GridSearchCV enumerates every θ in the Cartesian product
  of PARAM_GRID (exhaustive search) and picks:

    θ* = argmin_θ CV_score(θ)

  In this run:   |Θ| = 3 × 3 × 3 = 27 candidates
                 Total model fits = 27 × 3 = 81

  ┌────────────────────────────────────────────────────────────┐
  │ 2. WHY CROSS-VALIDATION IS NEEDED                          │
  └────────────────────────────────────────────────────────────┘
  Evaluating on the same data used to choose θ is optimistic:

    • A single held-out set is a RANDOM split — its score has
      high variance depending on which samples land in it.
    • k-fold CV uses EVERY sample for validation exactly once,
      giving a lower-variance, nearly unbiased estimate of
      generalisation error.
    • Without CV, we would over-select hyperparameters that
      happen to suit the particular random test split →
      the tuned model may perform worse on real new data.

  ┌────────────────────────────────────────────────────────────┐
  │ 3. BIAS-VARIANCE TRADEOFF IN HYPERPARAMETER TUNING         │
  └────────────────────────────────────────────────────────────┘
  For SVR the key knobs are C and epsilon (ε):

  ┌────────┬────────────────┬────────────────┬───────────────┐
  │ Param  │ Low value      │ High value     │ Effect        │
  ├────────┼────────────────┼────────────────┼───────────────┤
  │ C      │ Wide margin,   │ Narrow margin, │ Low C → high  │
  │        │ many violations│ few violations │ bias / under- │
  │        │ allowed        │ (fits noise)   │ fit risk      │
  ├────────┼────────────────┼────────────────┼───────────────┤
  │ ε      │ Tight tube,    │ Wide tube,     │ Large ε →     │
  │        │ more support   │ fewer SVs,     │ smoother fit, │
  │        │ vectors        │ simpler model  │ higher bias   │
  ├────────┼────────────────┼────────────────┼───────────────┤
  │ gamma  │ Broad Gaussian │ Narrow Gaussian│ High γ →      │
  │ (RBF)  │ → smoother     │ → wiggly fit   │ overfit risk  │
  └────────┴────────────────┴────────────────┴───────────────┘

  The sweet spot (θ*) minimises the total error:

     Total Error = Bias² + Variance + Irreducible Noise

  GridSearchCV finds this empirically via CV without assuming
  any parametric form for the error surface.

  ┌────────────────────────────────────────────────────────────┐
  │ 4. DATA LEAKAGE PREVENTION                                 │
  └────────────────────────────────────────────────────────────┘
  The StandardScaler is INSIDE the Pipeline:

    Pipeline(scaler → SVR)

  GridSearchCV calls pipeline.fit(X_train_fold) for each fold,
  so the scaler ONLY sees training fold data, never the
  validation fold. This is critical — fitting the scaler on
  the entire dataset before CV would leak test distribution
  information and make CV scores over-optimistic.
    """)
    print(text)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + SEP)
    print("  🌱  PRECISION AGRICULTURE — SVR HYPERPARAMETER TUNING")
    print(SEP)

    np.random.seed(RANDOM_STATE)

    # ── Shared data prep (reuse functions from svr_model) ─
    df              = load_data(DATA_PATH)
    df_enc, encoders = encode_categoricals(df, CAT_COLS)
    X, y            = split_features_target(df_enc, TARGET_COL, DROP_COLS)
    X_train, X_test, y_train, y_test = split_and_subsample(
        X, y, SUBSAMPLE_N, TEST_SIZE, RANDOM_STATE
    )

    # ── Re-run all original kernels for comparison table ──
    from svr_model import train_all_kernels
    all_kernel_results = train_all_kernels(
        X_train, X_test, y_train, y_test, SVR_CONFIGS
    )

    # ── Step 1: Default RBF baseline ──────────────────────
    default_result = train_default_rbf(X_train, X_test, y_train, y_test)

    # ── Step 2: GridSearchCV ──────────────────────────────
    grid_search = run_grid_search(
        X_train, y_train,
        param_grid       = PARAM_GRID,
        cv               = CV_FOLDS,
        n_jobs           = N_JOBS,
        tuning_subsample = TUNING_SUBSAMPLE,
        random_state     = RANDOM_STATE,
    )

    # ── Step 3: Retrain with best params ──────────────────
    tuned_result = retrain_tuned_rbf(
        grid_search, X_train, X_test, y_train, y_test
    )

    # ── Step 4: CV heatmap ────────────────────────────────
    plot_cv_heatmap(grid_search)

    # ── Step 5: Comparison plot ───────────────────────────
    plot_comparison(default_result, tuned_result, y_test, PLOT_N)

    # ── Step 6: Updated results table ────────────────────
    print_results_table(default_result, tuned_result, all_kernel_results)

    # ── Step 7: Theory ────────────────────────────────────
    print_theory()

    # ── Step 8: Export Model & Encoders ────────────────
    _sec("STEP 8 — EXPORTING TUNED MODEL & ENCODERS")
    model_path = Path("project/models/best_svr_model.pkl")
    encoder_path = Path("project/models/encoders.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the pipeline
    joblib.dump(tuned_result["pipeline"], model_path)
    
    # Save the encoders
    joblib.dump(encoders, encoder_path)
    print(f"  💾  Best model saved to → '{model_path}'")
    print(f"  💾  Encoders saved to   → '{encoder_path}'")

    print(f"\n{SEP}")
    print("  ✅  TUNING PIPELINE COMPLETE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()

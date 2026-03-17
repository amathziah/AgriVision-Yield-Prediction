"""
==============================================================
  Satellite-Based Precision Agriculture
  Support Vector Regression (SVR) — Multi-Kernel Pipeline
==============================================================
Input  : project/data/tabular/final_dataset.csv
Output : project/results/svr_*.png  +  console metrics table

Kernels compared
  • Linear      — fast, interpretable, good baseline
  • RBF         — captures non-linear patterns, usually best
  • Polynomial  — models interaction terms explicitly

Dependencies: pandas, numpy, scikit-learn, matplotlib, tabulate
"""

import os
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_PATH    = BASE_DIR / "project" / "data" / "tabular" / "final_dataset.csv"
RESULTS_DIR  = BASE_DIR / "project" / "results"

TARGET_COL   = "yield_tonnes_per_ha"
CAT_COLS     = ["country", "crop"]
DROP_COLS    = ["year"]          # year leaks temporal ordering — exclude

TEST_SIZE    = 0.20
RANDOM_STATE = 42

# SVR hyperparameters (production sensible defaults)
SVR_CONFIGS: dict[str, dict] = {
    "Linear": dict(
        kernel="linear",
        C=1.0,
        epsilon=0.1,
        max_iter=5000,
    ),
    "RBF": dict(
        kernel="rbf",
        C=10.0,
        epsilon=0.1,
        gamma="scale",
        max_iter=5000,
    ),
    "Polynomial": dict(
        kernel="poly",
        C=1.0,
        epsilon=0.1,
        degree=3,
        coef0=1,
        gamma="scale",
        max_iter=5000,
    ),
}

# Subsample for speed (SVR is O(n²) – use full data only for final training)
SUBSAMPLE_N  = 8_000
PLOT_N       = 500          # points shown in the Predicted vs Actual scatter

SEP = "=" * 64


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def _sec(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _ensure_results() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
# STEP 1 — LOAD & PREVIEW
# ──────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    _sec("STEP 1 — LOADING DATASET")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run src/merge_datasets.py first."
        )

    df = pd.read_csv(path)
    print(f"  ✅  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns  : {list(df.columns)}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────
# STEP 2 — ENCODE CATEGORICALS
# ──────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode each categorical column.

    Label encoding (integer codes) is appropriate here because:
      • SVR's kernel functions operate on numeric distances/dot-products.
      • One-hot encoding 212 countries → 212 sparse binary columns would
        explode dimensionality and hurt RBF kernel distance calculations.
      • The ordinal integer representation is then removed by StandardScaler
        (mean=0, std=1), so position in sorted order has no undue influence.

    Returns
    -------
    df_encoded : DataFrame with encoded columns
    encoders   : dict mapping column name → fitted LabelEncoder
                 (needed to inverse-transform predictions later)
    """
    _sec("STEP 2 — ENCODING CATEGORICAL VARIABLES")

    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in cat_cols:
        if col not in df.columns:
            print(f"  ⚠️  Column '{col}' not found — skipping.")
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  ✅  '{col}' encoded → {len(le.classes_)} unique classes")

    return df, encoders


# ──────────────────────────────────────────────────────────
# STEP 3 — FEATURE / TARGET SPLIT
# ──────────────────────────────────────────────────────────

def split_features_target(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X from the target vector y.
    Drops any columns listed in drop_cols (e.g. 'year').
    """
    _sec("STEP 3 — FEATURE / TARGET SPLIT")

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target_col] + cols_to_drop)
    y = df[target_col]

    print(f"  Feature columns : {list(X.columns)}")
    print(f"  Target column   : '{target_col}'")
    print(f"  X shape         : {X.shape}")
    print(f"  y shape         : {y.shape}")
    print(f"  y range         : [{y.min():.3f}, {y.max():.3f}]  "
          f"mean={y.mean():.3f}  std={y.std():.3f}")
    return X, y


# ──────────────────────────────────────────────────────────
# STEP 4 — TRAIN / TEST SPLIT  +  SUBSAMPLE
# ──────────────────────────────────────────────────────────

def split_and_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    subsample_n: int,
    test_size: float,
    random_state: int,
) -> tuple:
    """
    Subsample the training set to keep SVR training tractable
    (O(n²) time complexity), then split into train / test.

    The test set uses the same random split of the full dataset
    so evaluation reflects real generalisation performance.
    """
    _sec("STEP 4 — TRAIN / TEST SPLIT")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Subsample training rows for SVR scalability
    n = min(subsample_n, len(X_train_full))
    idx = np.random.RandomState(random_state).choice(
        len(X_train_full), size=n, replace=False
    )
    X_train = X_train_full.iloc[idx]
    y_train = y_train_full.iloc[idx]

    print(f"  Full dataset    : {len(X):>6} samples")
    print(f"  Train (sampled) : {len(X_train):>6} samples  "
          f"({100*len(X_train)/len(X):.1f}% of total)")
    print(f"  Test            : {len(X_test):>6} samples  "
          f"({100*len(X_test)/len(X):.1f}% of total)")

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────
# STEP 5 — TRAIN ALL SVR KERNELS
# ──────────────────────────────────────────────────────────

def train_all_kernels(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    svr_configs: dict[str, dict],
) -> dict[str, dict]:
    """
    For each kernel:
      1. Build an sklearn Pipeline: StandardScaler → SVR
      2. Fit on (X_train, y_train)
      3. Predict on X_test
      4. Compute MAE and RMSE

    WHY A PIPELINE?
    ────────────────
    StandardScaler must be fit ONLY on training data.
    sklearn Pipeline ensures the scaler never sees test data,
    preventing data leakage — a common source of over-optimistic metrics.

    WHY SCALING IS CRITICAL FOR SVR
    ────────────────────────────────
    SVR's kernel functions compute distances or dot-products between
    feature vectors.  Features on different scales (e.g. rainfall in
    hundreds vs temperature in tens) cause the kernel to be dominated
    by the larger-scaled feature, completely ignoring the smaller ones.
    StandardScaler brings every feature to μ=0, σ=1 so each dimension
    contributes equally to the kernel computation.

    Returns
    -------
    results : dict  {kernel_name → {"pipeline", "y_pred", "mae", "rmse"}}
    """
    _sec("STEP 5 — TRAINING SVR KERNELS")

    results: dict[str, dict] = {}

    for name, params in svr_configs.items():
        print(f"\n  🔧  [{name}] Kernel — params: {params}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svr",    SVR(**params)),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results[name] = {
            "pipeline": pipeline,
            "y_pred"  : y_pred,
            "mae"     : mae,
            "rmse"    : rmse,
        }

        print(f"       MAE  = {mae:.4f} tonnes/ha")
        print(f"       RMSE = {rmse:.4f} tonnes/ha")

    return results


# ──────────────────────────────────────────────────────────
# STEP 6 — PLOT PREDICTED VS ACTUAL
# ──────────────────────────────────────────────────────────

def plot_predicted_vs_actual(
    results: dict[str, dict],
    y_test: pd.Series,
    plot_n: int,
) -> None:
    """
    Create a 2×2 grid:
      • 3 scatter plots (one per kernel) — Predicted vs Actual
      • 1 residual plot (RBF) — Residuals vs Fitted
    """
    _ensure_results()

    n_kernels = len(results)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "SVR — Predicted vs Actual Crop Yield  (tonnes/ha)",
        fontsize=16, fontweight="bold", y=1.01
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    colors = {"Linear": "#4C72B0", "RBF": "#55A868", "Polynomial": "#C44E52"}

    idx = np.random.RandomState(42).choice(len(y_test), size=min(plot_n, len(y_test)), replace=False)
    y_sample = np.array(y_test)[idx]

    for ax_idx, (name, res) in enumerate(results.items()):
        ax = axes[ax_idx]
        y_pred_sample = res["y_pred"][idx]
        color = colors.get(name, "#888888")

        ax.scatter(y_sample, y_pred_sample,
                   alpha=0.45, s=20, color=color, edgecolors="none")

        # Perfect-prediction diagonal
        lims = [
            min(y_sample.min(), y_pred_sample.min()) - 0.5,
            max(y_sample.max(), y_pred_sample.max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", linewidth=1.2, label="Perfect fit")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("Actual yield (t/ha)", fontsize=10)
        ax.set_ylabel("Predicted yield (t/ha)", fontsize=10)
        ax.set_title(
            f"{name} Kernel\n"
            f"MAE={res['mae']:.3f}  RMSE={res['rmse']:.3f}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    # 4th panel: Residuals for best kernel (RBF)
    ax_res = axes[3]
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_pred = results[best_name]["y_pred"][idx]
    residuals  = y_sample - best_pred

    ax_res.scatter(best_pred, residuals,
                   alpha=0.45, s=20, color="#8172B2", edgecolors="none")
    ax_res.axhline(0, color="k", linewidth=1.2, linestyle="--")
    ax_res.set_xlabel("Fitted values", fontsize=10)
    ax_res.set_ylabel("Residuals", fontsize=10)
    ax_res.set_title(
        f"Residuals vs Fitted  [{best_name} Kernel — best RMSE]",
        fontsize=11, fontweight="bold"
    )
    ax_res.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_path = RESULTS_DIR / "svr_predicted_vs_actual.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊  Plot saved → '{save_path}'")
    plt.close()


# ──────────────────────────────────────────────────────────
# STEP 7 — KERNEL COMPARISON TABLE
# ──────────────────────────────────────────────────────────

def print_comparison_table(results: dict[str, dict]) -> None:
    _sec("STEP 7 — KERNEL COMPARISON TABLE")

    rows = []
    for name, res in results.items():
        rows.append({
            "Kernel"     : name,
            "MAE  (t/ha)": f"{res['mae']:.4f}",
            "RMSE (t/ha)": f"{res['rmse']:.4f}",
            "Rank (RMSE)": "",
        })

    # Rank by RMSE
    sorted_names = sorted(results, key=lambda k: results[k]["rmse"])
    rank_map = {name: i + 1 for i, name in enumerate(sorted_names)}
    medals   = {1: "🥇", 2: "🥈", 3: "🥉"}
    for row in rows:
        r = rank_map[row["Kernel"]]
        row["Rank (RMSE)"] = f"{medals.get(r, r)}  #{r}"

    df_table = pd.DataFrame(rows)
    print(df_table.to_string(index=False))

    best = sorted_names[0]
    print(f"\n  🏆  Best kernel: {best} "
          f"(RMSE = {results[best]['rmse']:.4f} t/ha, "
          f"MAE = {results[best]['mae']:.4f} t/ha)")


# ──────────────────────────────────────────────────────────
# STEP 8 — THEORY EXPLANATION
# ──────────────────────────────────────────────────────────

def print_explanation() -> None:
    _sec("STEP 8 — THEORY: SVR INTERNALS")

    text = textwrap.dedent("""
  ┌──────────────────────────────────────────────────────────┐
  │ 1. THE KERNEL TRICK (Mathematical Intuition)             │
  └──────────────────────────────────────────────────────────┘
  SVR finds a hyperplane in feature space that fits the data
  within an ε-tube. For non-linear data, instead of computing
  explicit feature maps φ(x), the kernel trick replaces the
  dot product with a kernel function K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ).

  This allows SVR to operate implicitly in a very high-
  (or even infinite-) dimensional space without ever computing
  the mapping φ explicitly — massive computational savings.

  ┌──────────────────────────────────────────────────────────┐
  │ 2. KERNEL COMPARISON                                     │
  └──────────────────────────────────────────────────────────┘
  ┌─────────────┬─────────────────────────┬─────────────────┐
  │ Kernel      │ K(xᵢ, xⱼ)              │ Best for        │
  ├─────────────┼─────────────────────────┼─────────────────┤
  │ Linear      │ xᵢ · xⱼ                │ Linearly sepa-  │
  │             │                         │ rable, high-dim │
  │             │                         │ sparse data     │
  ├─────────────┼─────────────────────────┼─────────────────┤
  │ RBF         │ exp(−γ‖xᵢ−xⱼ‖²)        │ General-purpose │
  │ (Gaussian)  │                         │ unknown shape;  │
  │             │                         │ typically best  │
  ├─────────────┼─────────────────────────┼─────────────────┤
  │ Polynomial  │ (γ xᵢ·xⱼ + r)^d        │ Interaction     │
  │             │                         │ terms; NLP-like │
  │             │                         │ features        │
  └─────────────┴─────────────────────────┴─────────────────┘
  γ = gamma,  r = coef0,  d = degree

  ┌──────────────────────────────────────────────────────────┐
  │ 3. ROLE OF HYPERPARAMETERS C AND ε (epsilon)             │
  └──────────────────────────────────────────────────────────┘
  C — Regularisation strength
    • Low C  → wide margin, more violations allowed → underfitting risk
    • High C → narrow margin, fewer violations → overfitting risk
    • Think of C as the "penalty per violation" of the ε-tube

  ε (epsilon) — Tube half-width
    • Predictions inside ±ε of true value incur ZERO loss
    • Controls how sensitive the model is to small errors
    • Large ε → simpler model; small ε → tighter fit, more support vectors
    • Rule of thumb: set ε ≈ 10% of the target std deviation

  ┌──────────────────────────────────────────────────────────┐
  │ 4. WHY SCALING IS MANDATORY FOR SVR                      │
  └──────────────────────────────────────────────────────────┘
  SVR computes K(xᵢ, xⱼ) which involves distances or dot-products.

  Example without scaling:
    rainfall ≈ 1200  |  temperature ≈ 21  |  pesticides ≈ 5000

  Without scaling, K is dominated by pesticides (5000² term)
  while temperature contributes almost nothing (21²).

  StandardScaler transforms each feature xₖ → (xₖ − μₖ) / σₖ
  → all features have mean=0, variance=1 → equal kernel influence.
    """)
    print(text)


# ──────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + SEP)
    print("  🌱  PRECISION AGRICULTURE — SVR MULTI-KERNEL PIPELINE")
    print(SEP)

    np.random.seed(RANDOM_STATE)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Encode categoricals
    df_enc, encoders = encode_categoricals(df, CAT_COLS)

    # 3. Split features / target
    X, y = split_features_target(df_enc, TARGET_COL, DROP_COLS)

    # 4. Train / test split + subsample
    X_train, X_test, y_train, y_test = split_and_subsample(
        X, y, SUBSAMPLE_N, TEST_SIZE, RANDOM_STATE
    )

    # 5. Train all kernels
    results = train_all_kernels(
        X_train, X_test, y_train, y_test, SVR_CONFIGS
    )

    # 6. Plot
    plot_predicted_vs_actual(results, y_test, PLOT_N)

    # 7. Comparison table
    print_comparison_table(results)

    # 8. Theory explanation
    print_explanation()

    print(f"\n{SEP}")
    print("  ✅  SVR PIPELINE COMPLETE")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()

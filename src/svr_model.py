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
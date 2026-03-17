"""
=============================================================
Satellite-Based Precision Agriculture
Data Preprocessing Pipeline
=============================================================
Steps:
    1. Load CSV dataset
    2. Basic Exploratory Data Analysis (EDA)
    3. Handle missing values
    4. Normalize / scale features

Dependencies: pandas, numpy, scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import os
import sys


# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

# Path to your CSV file – update this before running
CSV_PATH = "dataset/agriculture_data.csv"

# Target column (not to be scaled)
TARGET_COLUMN = "crop_yield"

# Columns to drop outright (e.g., identifiers, free-text)
COLUMNS_TO_DROP: list[str] = []

# Strategy for imputing numerical missing values:
#   "mean" | "median" | "most_frequent" | "constant"
NUMERICAL_IMPUTE_STRATEGY = "median"

# Strategy for imputing categorical missing values:
#   "most_frequent" | "constant"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

# Scaler to use for numerical features:
#   "minmax"   -> scales to [0, 1]
#   "standard" -> zero mean, unit variance (Z-score)
SCALER_TYPE = "minmax"

# If True, the cleaned / scaled DataFrame is saved to disk
SAVE_OUTPUT = True
OUTPUT_PATH = "dataset/processed_agriculture_data.csv"


# ──────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please update CSV_PATH in the CONFIGURATION section."
        )

    df = pd.read_csv(path)
    print(f"✅ Dataset loaded from '{path}'")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    return df


# ──────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """
    Print a concise EDA summary:
        - First 5 rows
        - Column dtypes & non-null counts
        - Missing value report
        - Basic descriptive statistics

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe to analyse.
    """
    separator = "=" * 60

    # --- Head ---
    print(separator)
    print("📋  FIRST 5 ROWS")
    print(separator)
    print(df.head().to_string())
    print()

    # --- Info ---
    print(separator)
    print("ℹ️   DATAFRAME INFO")
    print(separator)
    df.info()
    print()

    # --- Missing values ---
    print(separator)
    print("❓  MISSING VALUES")
    print(separator)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })
    missing_report = missing_report[missing_report["Missing Count"] > 0]

    if missing_report.empty:
        print("   No missing values found — dataset is complete. ✔️")
    else:
        print(missing_report.to_string())
    print()

    # --- Descriptive statistics ---
    print(separator)
    print("📊  DESCRIPTIVE STATISTICS (numerical columns)")
    print(separator)
    print(df.describe().round(4).to_string())
    print()


# ──────────────────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ──────────────────────────────────────────────────────────

def handle_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """
    Impute missing values separately for numerical and categorical columns.

    Numerical  → filled with the column median (or mean / mode, configurable)
    Categorical → filled with the most frequent value (mode)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that may contain NaN values.
    numerical_strategy : str
        Imputation strategy for numerical columns.
    categorical_strategy : str
        Imputation strategy for categorical columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with no missing values.
    """
    df = df.copy()

    # Separate column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- Impute numerical columns ---
    if numerical_cols:
        num_imputer = SimpleImputer(strategy=numerical_strategy)
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        print(f"✅ Numerical columns imputed using strategy='{numerical_strategy}'")
        print(f"   Columns: {numerical_cols}\n")

    # --- Impute categorical columns ---
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        print(f"✅ Categorical columns imputed using strategy='{categorical_strategy}'")
        print(f"   Columns: {categorical_cols}\n")

    # Confirm no nulls remain
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls == 0:
        print("✔️  No missing values remain after imputation.\n")
    else:
        print(f"⚠️  Warning: {remaining_nulls} null value(s) still present.\n")

    return df


# ──────────────────────────────────────────────────────────
# 4. NORMALIZE / SCALE FEATURES
# ──────────────────────────────────────────────────────────

def normalize_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    scaler_type: str = "minmax",
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler]:
    """
    Scale all numerical feature columns (excluding the target column).

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataframe (no missing values).
    target_col : str
        Name of the target/label column to exclude from scaling.
    scaler_type : str
        "minmax" for MinMaxScaler or "standard" for StandardScaler.

    Returns
    -------
    tuple[pd.DataFrame, scaler]
        - Dataframe with scaled features.
        - Fitted scaler object (needed to inverse-transform predictions later).
    """
    df = df.copy()

    # Identify feature columns (numerical, excluding target)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numerical_cols if c != target_col]

    if not feature_cols:
        print("⚠️  No numerical feature columns found to scale.\n")
        return df, None

    # Instantiate the chosen scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
        scaler_label = "MinMaxScaler  [0, 1]"
    elif scaler_type == "standard":
        scaler = StandardScaler()
        scaler_label = "StandardScaler  (μ=0, σ=1)"
    else:
        raise ValueError(
            f"Unknown scaler_type='{scaler_type}'. Choose 'minmax' or 'standard'."
        )

    # Fit & transform
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"✅ Features normalized using {scaler_label}")
    print(f"   Scaled columns : {feature_cols}")
    print(f"   Target column  : '{target_col}' (left unchanged)\n")

    return df, scaler


# ──────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def run_pipeline(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline end-to-end:
        Step 1 → Load
        Step 2 → EDA
        Step 3 → Impute missing values
        Step 4 → Scale features

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe, ready for model training.
    """
    print("\n" + "=" * 60)
    print("  🌱  PRECISION AGRICULTURE — PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load
    df_raw = load_dataset(csv_path)

    # Drop unwanted columns if configured
    if COLUMNS_TO_DROP:
        df_raw.drop(
            columns=[c for c in COLUMNS_TO_DROP if c in df_raw.columns],
            inplace=True,
        )
        print(f"🗑️  Dropped columns: {COLUMNS_TO_DROP}\n")

    # Step 2: EDA
    run_eda(df_raw)

    # Step 3: Handle missing values
    print("=" * 60)
    print("  STEP 3 — HANDLING MISSING VALUES")
    print("=" * 60 + "\n")
    df_clean = handle_missing_values(
        df_raw,
        numerical_strategy=NUMERICAL_IMPUTE_STRATEGY,
        categorical_strategy=CATEGORICAL_IMPUTE_STRATEGY,
    )

    # Step 4: Normalize
    print("=" * 60)
    print("  STEP 4 — FEATURE NORMALIZATION")
    print("=" * 60 + "\n")
    df_processed, fitted_scaler = normalize_features(
        df_clean,
        target_col=TARGET_COLUMN,
        scaler_type=SCALER_TYPE,
    )

    # Preview processed output
    print("=" * 60)
    print("  ✅  PREPROCESSED DATA — PREVIEW")
    print("=" * 60)
    print(df_processed.head().to_string())
    print(f"\n   Final shape: {df_processed.shape[0]} rows × {df_processed.shape[1]} columns\n")

    # Optionally save to disk
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df_processed.to_csv(OUTPUT_PATH, index=False)
        print(f"💾  Processed dataset saved to '{OUTPUT_PATH}'\n")

    return df_processed


# ──────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optionally accept CSV path as a CLI argument:
    #   python data_preprocessing.py path/to/your_data.csv
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    processed_df = run_pipeline(csv_path=path)

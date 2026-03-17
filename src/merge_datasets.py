"""
==============================================================
  Satellite-Based Precision Agriculture
  Multi-CSV Merge Pipeline
==============================================================
Input files
  ┌──────────────────┬─────────────────────────────────────┐
  │ rainfall.csv     │ Area, Year, average_rain_fall_mm_per_year │
  │ temp.csv         │ year, country, avg_temp             │
  │ pesticides.csv   │ Domain, Area, Element, Item, Year,  │
  │                  │ Unit, Value                         │
  │ yield.csv        │ Domain Code, Domain, Area Code,     │
  │                  │ Area, Element Code, Element,        │
  │                  │ Item Code, Item, Year Code, Year,   │
  │                  │ Unit, Value                         │
  └──────────────────┴─────────────────────────────────────┘

Join strategy
  • All datasets share (country, year) as natural keys.
  • yield.csv also carries a Crop dimension → three-way key
    (country, year, crop) after pesticides is broadcast per crop.
  • We use LEFT JOIN from yield outward so every crop-level
    observation keeps its climate/soil context even when some
    auxiliary records are absent.

Output
  project/data/tabular/final_dataset.csv   ← ML-ready

Dependencies: pandas, numpy
"""

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent   # project root
TABULAR    = BASE_DIR / "project" / "data" / "tabular"
OUTPUT_DIR = TABULAR                                   # save alongside source files

FILE_PATHS = {
    "rainfall"   : TABULAR / "rainfall.csv",
    "temp"       : TABULAR / "temp.csv",
    "pesticides" : TABULAR / "pesticides.csv",
    "yield"      : TABULAR / "yield.csv",
}

OUTPUT_PATH = OUTPUT_DIR / "final_dataset.csv"

SEP = "=" * 64


# ──────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _preview(df: pd.DataFrame, label: str, n: int = 3) -> None:
    print(f"\n  [{label}]  shape={df.shape}")
    print(df.head(n).to_string(index=False))


def _missing_report(df: pd.DataFrame, label: str) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"  ✔  [{label}] No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({"Missing #": missing, "Missing %": pct})
        print(f"\n  ❓ [{label}] Missing values:\n{report.to_string()}")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace & lowercase all column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ──────────────────────────────────────────────────────────
# STEP 1 — LOAD ALL CSV FILES
# ──────────────────────────────────────────────────────────

def load_all(file_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """
    Load each CSV into a DataFrame, normalise column names,
    and print a quick inspection of what arrived.
    """
    _section("STEP 1 — LOADING CSV FILES")
    frames: dict[str, pd.DataFrame] = {}

    for name, path in file_paths.items():
        if not path.exists():
            print(f"  ⚠️  '{name}' not found at {path} — skipping.")
            continue
        df = pd.read_csv(path)
        df = _normalise_columns(df)
        size_mb = path.stat().st_size / 1_048_576
        frames[name] = df
        print(f"\n  ✅  {name:<12} loaded  "
              f"[{df.shape[0]:>6} rows × {df.shape[1]} cols | {size_mb:.2f} MB]")
        print(f"      Columns: {list(df.columns)}")

    return frames


# ──────────────────────────────────────────────────────────
# STEP 2 — INSPECT & FIND COMMON KEYS
# ──────────────────────────────────────────────────────────

def inspect_keys(frames: dict[str, pd.DataFrame]) -> None:
    """
    Print all columns per dataframe and highlight shared key candidates.
    """
    _section("STEP 2 — COLUMN INSPECTION & COMMON KEY DETECTION")

    all_cols: dict[str, set[str]] = {
        name: set(df.columns) for name, df in frames.items()
    }

    # Find columns common to ALL loaded frames
    common = set.intersection(*all_cols.values()) if all_cols else set()
    print(f"\n  Columns per file:")
    for name, cols in all_cols.items():
        print(f"    {name:<12}: {sorted(cols)}")

    print(f"\n  🔑 Common columns across ALL files: {sorted(common)}")
    print(
        "\n  Strategy:\n"
        "    • 'area'  ≈ country identifier  (renamed → 'country')\n"
        "    • 'year'  ≈ time key\n"
        "    • 'item'  ≈ crop type          (present in yield & pesticides)\n"
        "    → Three-way join key: (country, year, crop)"
    )


# ──────────────────────────────────────────────────────────
# STEP 3 — CLEAN INDIVIDUAL DATAFRAMES
# ──────────────────────────────────────────────────────────

def clean_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the relevant columns from yield.csv and rename them.

    Raw columns (normalised):
        domain_code, domain, area_code, area, element_code, element,
        item_code, item, year_code, year, unit, value
    We keep: area, item, year, value  → yield_hg_per_ha
    """
    keep = ["area", "item", "year", "value"]
    df = df[keep].copy()
    df.rename(columns={"area": "country", "item": "crop",
                        "value": "yield_hg_per_ha"}, inplace=True)
    # Remove rows where the yield value is missing
    df.dropna(subset=["yield_hg_per_ha"], inplace=True)
    df["year"] = df["year"].astype(int)
    return df


def clean_rainfall(df: pd.DataFrame) -> pd.DataFrame:
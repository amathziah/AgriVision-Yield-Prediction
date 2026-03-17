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
    """
    Raw columns (normalised): _area (leading space), year,
    average_rain_fall_mm_per_year

    The leading-space column name becomes 'area' after normalisation,
    but guard against edge cases.
    """
    # Detect the country column (may still have a leading underscore)
    country_col = next(
        (c for c in df.columns if "area" in c or "country" in c), None
    )
    if country_col is None:
        raise KeyError(f"Cannot find a country column in rainfall. Got: {list(df.columns)}")

    df = df[[country_col, "year", "average_rain_fall_mm_per_year"]].copy()
    df.rename(columns={country_col: "country"}, inplace=True)
    df["country"] = df["country"].str.strip()
    df["year"]    = df["year"].astype(int)
    # ⚡ Rainfall values are sometimes read as strings — force numeric
    df["average_rain_fall_mm_per_year"] = pd.to_numeric(
        df["average_rain_fall_mm_per_year"], errors="coerce"
    )
    return df


def clean_temp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw columns (normalised): year, country, avg_temp
    """
    df = df[["country", "year", "avg_temp"]].copy()
    df["country"] = df["country"].str.strip()
    df["year"]    = df["year"].astype(int)

    # temp.csv spans 1849-present; keep only years plausible for agriculture data
    df = df[df["year"] >= 1960].copy()

    # Average the temperature if multiple readings exist per (country, year)
    df = (df.groupby(["country", "year"], as_index=False)["avg_temp"]
            .mean()
            .round(4))
    return df


def clean_pesticides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw columns (normalised):
        domain, area, element, item, year, unit, value

    'value' = total pesticide use in tonnes of active ingredients.
    Pesticides are reported at (country, year) level — NOT per crop.
    We broadcast this value to every crop later via the join.
    """
    df = df[["area", "year", "value"]].copy()
    df.rename(columns={"area": "country", "value": "pesticides_tonnes"},
              inplace=True)
    df["country"] = df["country"].str.strip()
    df["year"]    = df["year"].astype(int)
    # If there are duplicate (country, year) entries, take the sum
    df = (df.groupby(["country", "year"], as_index=False)["pesticides_tonnes"]
            .sum())
    return df


# ──────────────────────────────────────────────────────────
# STEP 4 — MERGE ALL DATASETS
# ──────────────────────────────────────────────────────────

def merge_all(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge strategy
    ──────────────
    Base table : yield     (country, year, crop)  → yield_hg_per_ha
    Left join 1: rainfall  (country, year)         → average_rain_fall_mm_per_year
    Left join 2: temp      (country, year)         → avg_temp
    Left join 3: pesticides(country, year)         → pesticides_tonnes

    JOIN TYPE: LEFT JOIN
    ────────────────────
    Why LEFT (not INNER)?
      • INNER JOIN drops any crop-year observation that lacks a
        matching climate record. Countries with shorter climate
        histories would lose valid yield rows — introducing
        survivorship bias into the training data.
      • LEFT JOIN keeps ALL yield observations as the primary
        signal and fills missing climate values with NaN, which
        we handle explicitly in the post-merge cleaning step.

    Risks of wrong join choice
    ──────────────────────────
      INNER  → Silently drops rows → biased, shrunken dataset
      RIGHT  → Keeps all climate rows even with no yield data
               → meaningless rows flood the dataset
      OUTER  → Combines both bad sides → maximum noise
      LEFT   → ✅ Correct: yield drives the row set
    """
    _section("STEP 3 & 4 — CLEANING + MERGING DATASETS")

    # Validate all required files are present
    required = ["yield", "rainfall", "temp", "pesticides"]
    for key in required:
        if key not in frames:
            raise RuntimeError(
                f"Required file '{key}' was not loaded. "
                "Check FILE_PATHS in the configuration."
            )

    # ── Clean each frame ──────────────────────────────────
    df_yield      = clean_yield(frames["yield"])
    df_rainfall   = clean_rainfall(frames["rainfall"])
    df_temp       = clean_temp(frames["temp"])
    df_pesticides = clean_pesticides(frames["pesticides"])

    _preview(df_yield,      "yield (cleaned)")
    _preview(df_rainfall,   "rainfall (cleaned)")
    _preview(df_temp,       "temp (cleaned)")
    _preview(df_pesticides, "pesticides (cleaned)")

    # ── Merge ─────────────────────────────────────────────
    print(f"\n  🔗 Merging on (country, year) using LEFT JOIN ...")

    merged = df_yield.copy()
    initial_rows = len(merged)

    # LEFT JOIN 1: rainfall
    merged = merged.merge(df_rainfall, on=["country", "year"], how="left")
    print(f"      After + rainfall   : {len(merged):>7} rows")

    # LEFT JOIN 2: temperature
    merged = merged.merge(df_temp, on=["country", "year"], how="left")
    print(f"      After + temp       : {len(merged):>7} rows")

    # LEFT JOIN 3: pesticides (country + year only — broadcast across crops)
    merged = merged.merge(df_pesticides, on=["country", "year"], how="left")
    print(f"      After + pesticides : {len(merged):>7} rows")

    print(f"\n  ✅ Merge complete.  {initial_rows} → {len(merged)} rows "
          f"(difference due to duplicate year keys in auxiliary files)")

    return merged


# ──────────────────────────────────────────────────────────
# STEP 5 — POST-MERGE CLEANING
# ──────────────────────────────────────────────────────────

def post_merge_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    After the LEFT JOIN, NaN values appear where auxiliary
    datasets had no matching (country, year) record.

    Strategy
    ─────────
    • Numeric columns → fill with MEDIAN grouped by crop type.
      (Crop physiology means rice needs more rain than wheat —
       a global median would be misleading.)
    • Any row still missing yield_hg_per_ha (the target) → drop.
    • Categorical columns (country, crop) → already complete
      because they come from the left (yield) table.
    """
    _section("STEP 5 — POST-MERGE CLEANING")

    _missing_report(df, "merged (before clean)")

    numeric_fill_cols = [
        "average_rain_fall_mm_per_year",
        "avg_temp",
        "pesticides_tonnes",
    ]

    def _group_median_fill(series: pd.Series) -> pd.Series:
        """Coerce to float, then fill NaN with the group median."""
        s = pd.to_numeric(series, errors="coerce")
        return s.fillna(s.median())

    for col in numeric_fill_cols:
        if col not in df.columns:
            continue
        # Coerce the whole column to float FIRST (handles string leftovers)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Group-median fill: more ecologically meaningful than global median
        df[col] = df.groupby("crop")[col].transform(_group_median_fill)
        # Fallback: global median for crops whose entire group is NaN
        df[col] = df[col].fillna(df[col].median())

    # Drop rows where the TARGET is still NaN
    before = len(df)
    df.dropna(subset=["yield_hg_per_ha"], inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"  🗑️  Dropped {dropped} rows with missing target (yield_hg_per_ha).")

    _missing_report(df, "merged (after clean)")
    return df


# ──────────────────────────────────────────────────────────
# STEP 6 — FINAL CHECKS & SAVE
# ──────────────────────────────────────────────────────────

def finalise_and_save(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Enforce correct dtypes, reset the index, and save the ML-ready CSV.
    """
    _section("STEP 6 — FINAL DATASET & SAVE")

    # ── Dtype enforcement ─────────────────────────────────
    df["year"]    = df["year"].astype(int)
    df["country"] = df["country"].astype(str).str.strip().str.title()
    df["crop"]    = df["crop"].astype(str).str.strip().str.title()

    # Convert yield from hg/ha → tonnes/ha for human readability
    # 1 hg/ha = 0.0001 t/ha
    df["yield_tonnes_per_ha"] = (df["yield_hg_per_ha"] * 0.0001).round(4)
    df.drop(columns=["yield_hg_per_ha"], inplace=True)

    # ── Reset index ───────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    # ── Summary of final dataset ──────────────────────────
    print(f"\n  Final shape       : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns           : {list(df.columns)}")
    print(f"  Year range        : {df['year'].min()} – {df['year'].max()}")
    print(f"  Unique countries  : {df['country'].nunique()}")
    print(f"  Unique crops      : {df['crop'].nunique()}")
    print(f"\n  Feature summary:")
    print(df.describe(include="all").round(3).to_string())

    print(f"\n  First 5 rows of final dataset:")
    print(df.head().to_string(index=False))

    # ── Save ──────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"\n  💾  Saved → '{output_path}'  [{size_mb:.2f} MB]")

    return df


# ──────────────────────────────────────────────────────────
# STEP 7 — EXPLANATION (printed to console)
# ──────────────────────────────────────────────────────────

def print_explanation() -> None:
    _section("STEP 7 — ARCHITECTURE EXPLANATION")
    explanation = textwrap.dedent("""
  WHY MERGING IS IMPORTANT
  ─────────────────────────
  Each raw CSV captures only one dimension of the agricultural
  picture.  In isolation:

    • yield.csv       → tells us WHAT was grown and HOW MUCH
    • rainfall.csv    → tells us HOW MUCH water fell
    • temp.csv        → tells us HOW HOT it was
    • pesticides.csv  → tells us WHAT was applied to the soil

  A model trained on yield alone has no explanatory power.
  Once merged, every row says: "In country X, year Y, crop C
  grew Z tonnes/ha given R mm rain, T °C and P tonnes pesticide."
  This is the minimal feature set for a supervised regressor.

  JOIN TYPE USED: LEFT JOIN (yield as base table)
  ────────────────────────────────────────────────
  yield.csv is the PRIMARY table because it defines the
  observation unit: (country, year, crop).

  Climate / input tables are SECONDARY — they add context.
  A LEFT JOIN ensures:

    ✅  All yield observations are retained
    ✅  Missing climate context → NaN (handled by imputation)
    ❌  No synthetic rows are created

  RISKS OF WRONG JOIN TYPE
  ─────────────────────────
  ┌─────────────┬──────────────────────────────────────────┐
  │ INNER JOIN  │ Drops any yield row without a matching   │
  │             │ climate record. Creates survivorship bias │
  │             │ (only well-documented countries survive). │
  ├─────────────┼──────────────────────────────────────────┤
  │ RIGHT JOIN  │ Keeps ALL climate rows even if there is  │
  │             │ no yield observation — floods the dataset │
  │             │ with meaningless NaN targets.             │
  ├─────────────┼──────────────────────────────────────────┤
  │ OUTER JOIN  │ Worst of both worlds. Maximises row count │
  │             │ but introduces noise and NaN targets      │
  │             │ everywhere.                               │
  ├─────────────┼──────────────────────────────────────────┤
  │ LEFT JOIN ✅│ Keeps yield as ground truth; fills missing│
  │             │ climate values via group-median imputation.│
  └─────────────┴──────────────────────────────────────────┘

  POST-MERGE IMPUTATION CHOICE
  ─────────────────────────────
  Missing values are filled with the MEDIAN grouped by CROP TYPE
  rather than a global statistic because:
    • Rice needs ~1200 mm rain; wheat needs ~450 mm.
    • A global median (~700 mm) would severely mismatch both.
    • Crop-group median respects biological and climatic reality.
    """)
    print(explanation)


# ──────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def main() -> pd.DataFrame:
    print("\n" + SEP)
    print("  🌱  PRECISION AGRICULTURE — MULTI-CSV MERGE PIPELINE")
    print(SEP)

    # 1. Load
    frames = load_all(FILE_PATHS)

    # 2. Inspect keys
    inspect_keys(frames)

    # 3+4. Clean & merge
    merged = merge_all(frames)

    # 5. Post-merge cleaning
    cleaned = post_merge_clean(merged)

    # 6. Finalise & save
    final_df = finalise_and_save(cleaned, OUTPUT_PATH)

    # 7. Explanation
    print_explanation()

    print(f"\n{SEP}")
    print("  ✅  MERGE PIPELINE COMPLETE")
    print(f"{SEP}\n")

    return final_df


if __name__ == "__main__":
    df = main()

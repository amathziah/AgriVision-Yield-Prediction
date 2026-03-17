"""
==============================================================
  Satellite-Based Precision Agriculture
  Dataset Inspector & Project Setup Script
==============================================================
Steps:
    1. Inspect dataset/  → list files, auto-detect CSV & images
    2. Analyse tabular data (shape, columns, dtypes, missing %)
    3. Analyse image data  (tree, count, class structure, samples)
    4. Reorganise into clean project structure
    5. Print full explanation of how each dataset feeds the models

Dependencies: os, glob, shutil, pandas, numpy, matplotlib
"""

import os
import glob
import shutil
import random
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ──────────────────────────────────────────────────────────
# CONFIGURATION  (only these two values may need changing)
# ──────────────────────────────────────────────────────────

# Root of the raw dataset folder
DATASET_ROOT = "dataset"

# Root of the reorganised project (created if missing)
PROJECT_ROOT = "project"

# Candidate names used to auto-detect the target column
TARGET_CANDIDATES = [
    "yield", "crop_yield", "production", "output",
    "value", "area_harvested", "crop_production",
]

# Image extensions treated as valid
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# How many random sample images to display (Step 3)
N_SAMPLE_IMAGES = 3

SEP = "=" * 64


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────

def _separator(title: str) -> None:
    """Print a section divider with a title."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _find_csv_files(root: str) -> list[Path]:
    """Recursively find all .csv files under *root*."""
    return sorted(Path(root).rglob("*.csv"))


def _find_image_files(root: str) -> list[Path]:
    """Recursively find all image files under *root*."""
    return [
        p for p in Path(root).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _find_image_dirs(root: str) -> list[Path]:
    """
    Return directories that directly contain image files.
    Used to distinguish class-structured vs flat layouts.
    """
    dirs_with_images: set[Path] = set()
    for img in _find_image_files(root):
        dirs_with_images.add(img.parent)
    return sorted(dirs_with_images)


def _print_tree(directory: str, prefix: str = "", max_depth: int = 3,
                _depth: int = 0) -> None:
    """Recursively print a Unix-tree style directory listing."""
    if _depth > max_depth:
        return
    entries = sorted(Path(directory).iterdir())
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            _print_tree(str(entry), prefix + extension, max_depth, _depth + 1)


def _auto_detect_target(df: pd.DataFrame) -> str | None:
    """
    Return the first column whose name (case-insensitive) matches
    TARGET_CANDIDATES, or None if no match is found.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_cols:
            return lower_cols[candidate]
    return None


def _ensure_dir(path: str | Path) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────
# STEP 1 — INSPECT DATASET FOLDER
# ──────────────────────────────────────────────────────────

def step1_inspect_dataset(root: str) -> tuple[list[Path], list[Path]]:
    """
    Scan *root* and report every file / sub-folder found.
    Automatically separates CSV files from image files.

    Returns
    -------
    (csv_files, image_files)
    """
    _separator("STEP 1 — INSPECTING DATASET FOLDER")

    if not os.path.exists(root):
        print(f"⚠️  '{root}' not found. Please place your dataset folder here.")
        return [], []

    print(f"\n📂  Tree view of '{root}/':\n")
    _print_tree(root)

    csv_files   = _find_csv_files(root)
    image_files = _find_image_files(root)

    print(f"\n✅  Auto-detected CSV files  ({len(csv_files)}):")
    for f in csv_files:
        size_mb = f.stat().st_size / 1_048_576
        print(f"     {f.relative_to(root)}  [{size_mb:.2f} MB]")

    print(f"\n✅  Auto-detected image files ({len(image_files)}):")
    if image_files:
        # Group by extension
        ext_counts: dict[str, int] = {}
        for img in image_files:
            ext_counts[img.suffix.lower()] = ext_counts.get(img.suffix.lower(), 0) + 1
        for ext, count in sorted(ext_counts.items()):
            print(f"     {ext}: {count} file(s)")
    else:
        print("     (none found — image dataset not yet present)")

    return csv_files, image_files


# ──────────────────────────────────────────────────────────
# STEP 2 — ANALYSE TABULAR DATA
# ──────────────────────────────────────────────────────────

def step2_analyse_tabular(csv_files: list[Path]) -> dict[str, pd.DataFrame]:
    """
    Load every detected CSV and print a structured EDA summary.

    Returns
    -------
    dict mapping filename → DataFrame (for later use)
    """
    _separator("STEP 2 — ANALYSING TABULAR DATA")

    dataframes: dict[str, pd.DataFrame] = {}

    if not csv_files:
        print("⚠️  No CSV files to analyse.")
        return dataframes

    for csv_path in csv_files:
        short_name = csv_path.name
        print(f"\n{'─' * 56}")
        print(f"  📄 File: {csv_path}")
        print(f"{'─' * 56}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  ❌ Could not read '{short_name}': {exc}")
            continue

        # ── Shape & columns ───────────────────────────────
        print(f"\n  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Columns : {list(df.columns)}")

        # ── First 5 rows ──────────────────────────────────
        print("\n  First 5 rows:")
        print(df.head().to_string(index=False))

        # ── Auto-detect target column ─────────────────────
        target = _auto_detect_target(df)
        if target:
            print(f"\n  🎯  Auto-detected target column : '{target}'")
        else:
            print("\n  ⚠️  Target column NOT auto-detected. "
                  "Update TARGET_CANDIDATES in the config.")

        # ── Column type breakdown ─────────────────────────
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        print(f"\n  Numerical columns  ({len(num_cols)}) : {num_cols}")
        print(f"  Categorical columns({len(cat_cols)}) : {cat_cols}")
        if date_cols:
            print(f"  DateTime columns   ({len(date_cols)}) : {date_cols}")

        # ── Missing values ────────────────────────────────
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("\n  ✔️  No missing values detected.")
        else:
            mv_df = pd.DataFrame({
                "Missing": missing,
                "% Missing": (missing / len(df) * 100).round(2)
            })
            print("\n  ❓ Missing values detected:")
            print(mv_df.to_string())

        # ── Descriptive stats ─────────────────────────────
        print("\n  Descriptive statistics:")
        print(df.describe(include="all").round(3).to_string())

        dataframes[short_name] = df

    return dataframes


# ──────────────────────────────────────────────────────────
# STEP 3 — ANALYSE IMAGE DATA
# ──────────────────────────────────────────────────────────

def step3_analyse_images(image_files: list[Path], root: str) -> None:
    """
    Print a tree of the image dataset, count images per class (if any),
    and display N_SAMPLE_IMAGES random samples using matplotlib.
    """
    _separator("STEP 3 — ANALYSING IMAGE DATA")

    if not image_files:
        print(
            "  ℹ️  No image files found in the dataset folder.\n"
            "  → Place your image dataset under 'dataset/' and re-run.\n"
            "  → Typical structure:\n"
            "       dataset/images/class_A/img1.jpg\n"
            "       dataset/images/class_B/img2.jpg"
        )
        return

    # ── Tree view ─────────────────────────────────────────
    print(f"\n  Total images found: {len(image_files)}")
    print(f"\n  📂  Image directory tree:\n")
    image_dirs = _find_image_dirs(root)
    for d in image_dirs:
        _print_tree(str(d), max_depth=1)

    # ── Supervised vs flat ────────────────────────────────
    # Supervised: images sit in named sub-folders (one per class)
    # Flat: all images live in a single directory
    unique_parent_dirs = {f.parent for f in image_files}
    is_supervised = len(unique_parent_dirs) > 1

    if is_supervised:
        print("\n  🗂️  Structure: CLASS-BASED (supervised learning ready)")
        print("\n  Class breakdown:")
        class_counts: dict[str, int] = {}
        for f in image_files:
            class_name = f.parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for cls, count in sorted(class_counts.items()):
            print(f"     {cls:<30} {count:>5} image(s)")
    else:
        print("\n  🗂️  Structure: FLAT (all images in one folder — "
              "unsupervised / unlabelled)")

    # ── Display random samples ────────────────────────────
    n = min(N_SAMPLE_IMAGES, len(image_files))
    samples = random.sample(image_files, n)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Random Sample Images from Dataset", fontsize=14, fontweight="bold")
    for ax, img_path in zip(axes, samples):
        try:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(f"{img_path.parent.name}/{img_path.name}", fontsize=9)
            ax.axis("off")
        except Exception as exc:
            ax.set_title(f"Error: {exc}", fontsize=7)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("results/sample_images.png", dpi=150, bbox_inches="tight")
    print(f"\n  📸  Sample image grid saved → 'results/sample_images.png'")
    plt.show()


# ──────────────────────────────────────────────────────────
# STEP 4 — REORGANISE PROJECT STRUCTURE
# ──────────────────────────────────────────────────────────

def step4_reorganise_project(
    csv_files: list[Path],
    image_files: list[Path],
    project_root: str = PROJECT_ROOT,
    dataset_root: str = DATASET_ROOT,
) -> None:
    """
    Move detected files into a clean, conventional project layout:

        project/
        ├── data/
        │   ├── tabular/    ← CSV files
        │   └── images/     ← image files (preserving class sub-folders)
        ├── notebooks/
        ├── src/
        ├── models/
        └── results/
    """
    _separator("STEP 4 — REORGANISING PROJECT STRUCTURE")

    # ── Create all target directories ─────────────────────
    dirs_to_create = [
        f"{project_root}/data/tabular",
        f"{project_root}/data/images",
        f"{project_root}/notebooks",
        f"{project_root}/src",
        f"{project_root}/models",
        f"{project_root}/results",
    ]
    for d in dirs_to_create:
        _ensure_dir(d)
        print(f"  📁  Created (or verified): {d}/")

    # ── Move CSV files → data/tabular/ ────────────────────
    print(f"\n  📄  Moving CSV file(s) → {project_root}/data/tabular/")
    for csv_path in csv_files:
        dest = Path(project_root) / "data" / "tabular" / csv_path.name
        if dest.exists():
            print(f"       ⏭  Already exists, skipping: {dest.name}")
            continue
        shutil.copy2(str(csv_path), str(dest))   # copy2 preserves metadata
        print(f"       ✅  {csv_path.name}  →  {dest}")

    # ── Move image files → data/images/ ───────────────────
    print(f"\n  🖼️   Moving image file(s) → {project_root}/data/images/")
    if not image_files:
        print("       (no image files to move)")
    else:
        for img_path in image_files:
            # Preserve class sub-folder structure:
            #   dataset/images/class_A/img.jpg  →  project/data/images/class_A/img.jpg
            try:
                # Compute relative path from the detected image root
                rel = img_path.relative_to(dataset_root)
            except ValueError:
                rel = Path(img_path.name)
            dest = Path(project_root) / "data" / "images" / rel
            _ensure_dir(dest.parent)
            if dest.exists():
                continue
            shutil.copy2(str(img_path), str(dest))
        print(f"       ✅  {len(image_files)} image(s) copied.")

    # ── Final tree ────────────────────────────────────────
    print(f"\n  📂  Resulting project layout:\n")
    _print_tree(project_root, max_depth=3)


# ──────────────────────────────────────────────────────────
# STEP 5 — EXPLANATION
# ──────────────────────────────────────────────────────────

def step5_explain() -> None:
    """Print a clear, structured explanation of the ML architecture."""
    _separator("STEP 5 — ARCHITECTURE EXPLANATION")

    explanation = textwrap.dedent("""
    ┌─────────────────────────────────────────────────────────┐
    │  1. TABULAR DATASET                                     │
    └─────────────────────────────────────────────────────────┘
    The CSV contains per-area agricultural statistics:
      • rainfall      – annual/seasonal precipitation (mm)
      • temperature   – mean air temperature (°C)
      • pesticides    – pesticide usage (tonnes/hectare)
      • crop_yield    – the TARGET we want to predict (tonnes/ha)

    Source: FAO / Satellite-inferred ground measurements.
    Each row represents one (country, year, crop) observation.

    ┌─────────────────────────────────────────────────────────┐
    │  2. IMAGE DATASET                                       │
    └─────────────────────────────────────────────────────────┘
    Satellite or drone imagery organised per class (crop type
    or land-use category).  Each image is a patch of agricultural
    land captured in RGB (or multispectral) bands.

    Structure (expected):
        images/
          healthy_wheat/   ← class directories
          diseased_rice/
          bare_soil/

    ┌─────────────────────────────────────────────────────────┐
    │  3. HOW TABULAR DATA FEEDS → SVM                        │
    └─────────────────────────────────────────────────────────┘
    Pipeline:
      CSV  →  Impute missing values  →  StandardScaler
           →  feature matrix X  +  target vector y
           →  train/test split (80/20)
           →  Support Vector Regressor (SVR, kernel='rbf')
           →  GridSearchCV for (C, epsilon, gamma)
           →  Evaluate: RMSE, R²

    Why SVM?
      • Works well on small-to-medium tabular datasets
      • Robust to outliers via the epsilon-insensitive loss
      • Kernel trick captures non-linear climate–yield relationships

    ┌─────────────────────────────────────────────────────────┐
    │  4. HOW IMAGE DATA FEEDS → CNN                          │
    └─────────────────────────────────────────────────────────┘
    Pipeline:
      Images  →  Resize (224 × 224)  →  Normalise [0, 1]
              →  Augmentation (flip, rotate, zoom)
              →  CNN architecture (e.g., EfficientNetB0 fine-tuned)
              →  Softmax output  →  Class label (crop / land type)
              →  Evaluate: accuracy, F1-score, confusion matrix

    Why CNN?
      • Automatically learns spatial features (textures, NDVI patterns)
      • Transfer learning (ImageNet weights) reduces data requirements
      • Outperforms hand-crafted features for image classification

    ┌─────────────────────────────────────────────────────────┐
    │  5. HYBRID MODEL — COMBINING BOTH                       │
    └─────────────────────────────────────────────────────────┘
    Architecture:

       [Climate / Soil tabular features]
               │
               ▼
          ┌─────────┐        ┌──────────────────────┐
          │  SVM /  │        │  CNN Feature Extractor│
          │  MLP    │        │  (penultimate layer)  │
          └────┬────┘        └──────────┬───────────┘
               │  Tabular embedding     │  Image embedding
               └──────────┬────────────┘
                           │
                    CONCATENATE
                           │
                    ┌──────▼──────┐
                    │  Dense MLP  │  (Fusion head)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Crop Yield  │  ← Final prediction
                    │ Regression  │
                    └─────────────┘

    Rationale:
      • Tabular features capture measurable agronomic conditions
      • CNN features capture *visual* land/crop state from imagery
      • Fusing both lets the model see WHAT grows AND under WHAT conditions
      • Expected improvement: 8–15% RMSE reduction vs single-modality

    Data flow phases:
      Phase 1 → Train SVM on tabular data independently
      Phase 2 → Train CNN on image data independently
      Phase 3 → Freeze both bases, train fusion head end-to-end
      Phase 4 → Fine-tune full pipeline jointly (optional)
    """)

    print(explanation)


# ──────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────

def main() -> None:
    """Run all five steps in sequence."""

    # Make sure results/ exists for image output
    _ensure_dir("results")

    print("\n" + SEP)
    print("  🌱  PRECISION AGRICULTURE — DATASET INSPECTOR & SETUP")
    print(SEP)

    # ── Step 1: Scan dataset folder ───────────────────────
    csv_files, image_files = step1_inspect_dataset(DATASET_ROOT)

    # ── Step 2: EDA on tabular CSVs ───────────────────────
    dataframes = step2_analyse_tabular(csv_files)

    # ── Step 3: Image dataset analysis ───────────────────
    step3_analyse_images(image_files, DATASET_ROOT)

    # ── Step 4: Reorganise project ────────────────────────
    step4_reorganise_project(csv_files, image_files)

    # ── Step 5: Architecture explanation ─────────────────
    step5_explain()

    print("\n" + SEP)
    print("  ✅  ALL STEPS COMPLETE")
    print(SEP + "\n")


if __name__ == "__main__":
    main()

# 🌱 AgriVision — Satellite-Based Precision Agriculture & Yield Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVR%20%7C%20GridSearchCV-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Project Overview

**AgriVision** is a production-grade Machine Learning pipeline for **crop yield prediction** using satellite-inferred agricultural datasets.

The project combines:
- **Tabular ML** (SVR with kernel comparison + GridSearchCV tuning) on climate & pesticide data
- **Deep Learning (CNN)** on satellite/drone imagery *(coming soon)*
- **Hybrid Fusion Model** combining both modalities *(planned)*

---

## 🗂️ Project Structure

```
ML-agriculture/
│
├── dataset/                        ← Raw input CSVs
│   └── archive2/                   ← Sub-datasets (rainfall, temp, etc.)
│
├── project/
│   ├── data/
│   │   ├── tabular/                ← Processed CSVs + final_dataset.csv
│   │   └── images/                 ← Image dataset (add here)
│   ├── notebooks/                  ← Jupyter notebooks
│   ├── src/                        ← Production Python source
│   ├── models/                     ← Saved model artefacts
│   └── results/                    ← Plots, metrics, outputs
│
├── src/
│   ├── merge_datasets.py           ← Multi-CSV merge pipeline
│   ├── svr_model.py                ← SVR (Linear / RBF / Polynomial)
│   └── svr_tuning.py              ← GridSearchCV hyperparameter tuning
│
├── data_preprocessing.py           ← Standalone EDA + preprocessing
├── inspect_and_setup.py            ← Dataset inspector + project setup
└── README.md
```

---

## 📦 Datasets Used

| File | Description | Rows |
|------|-------------|------|
| `yield.csv` | Crop yield (hg/ha) by country, year, crop | 56,717 |
| `rainfall.csv` | Annual rainfall (mm) by country, year | 6,727 |
| `temp.csv` | Average temperature (°C) by country, year | 71,311 |
| `pesticides.csv` | Pesticide usage (tonnes) by country, year | 4,349 |

**Merged output:** `project/data/tabular/final_dataset.csv`  
→ 56,717 rows × 7 columns, zero missing values after group-median imputation.

---

## ⚙️ Pipeline Steps

### 1. Data Merging (`src/merge_datasets.py`)
- Auto-detects CSV files and common join keys
- LEFT JOIN on `(country, year)` — yield as base table
- Group-median imputation for missing climate values
- Unit conversion: `hg/ha → tonnes/ha`

### 2. SVR Model (`src/svr_model.py`)
- Label-encodes `country` (212 classes) and `crop` (10 classes)
- sklearn `Pipeline`: `StandardScaler → SVR`
- Compares three kernels:

| Kernel | MAE (t/ha) | RMSE (t/ha) |
|--------|-----------|------------|
| RBF | **3.567** | **6.009** |
| Linear | 4.653 | 7.683 |
| Polynomial | 54.743 | 56.179 |

### 3. Hyperparameter Tuning (`src/svr_tuning.py`)
- `GridSearchCV` over `C × gamma × epsilon` (27 combos × 3 folds = **81 fits**)
- Best params: `C=100, gamma='scale', epsilon=1.0`

| Model | MAE (t/ha) | RMSE (t/ha) |
|-------|-----------|------------|
| RBF (default) | 3.567 | 6.009 |
| **RBF (tuned)** | **3.476** | **5.728** |

**Improvement: RMSE ↓ 4.7%**

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/amathziah/AgriVision-Yield-Prediction.git
cd AgriVision-Yield-Prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib

# 3. Merge raw CSVs into final dataset
python src/merge_datasets.py

# 4. Train and evaluate SVR kernels
python src/svr_model.py

# 5. Run hyperparameter tuning
python src/svr_tuning.py
```

---

## 📊 Results

All plots are saved to `project/results/`:

| File | Description |
|------|-------------|
| `svr_predicted_vs_actual.png` | Predicted vs Actual for all 3 kernels |
| `svr_gridsearch_heatmap.png` | CV RMSE heatmap across C × gamma |
| `svr_default_vs_tuned.png` | Default vs Tuned RBF comparison |

---

## 🔮 Planned Next Steps

- [ ] CNN pipeline for satellite image classification
- [ ] Hybrid fusion model (tabular SVR + CNN embeddings)
- [ ] Model serialisation & prediction API
- [ ] Streamlit dashboard

---

## 👤 Author

**Amathziah** — [GitHub](https://github.com/amathziah)

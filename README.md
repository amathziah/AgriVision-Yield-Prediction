# 🌱 AgriVision — Satellite-Based Precision Agriculture & Yield Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVR%20%7C%20GridSearchCV-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-emerald)

---

## 📌 Project Overview

**AgriVision** is a research-grade Hybrid Machine Learning pipeline for **crop yield prediction**. It achieves state-of-the-art results by fusing diverse data streams:
- **Tabular ML**: Support Vector Regression (SVR) tuned for climate & pesticide historical data.
- **Computer Vision**: Convolutional Neural Network (SatelliteCNN) for real-time visual crop health assessment.
- **Multi-Modal Fusion**: A weighted ensemble (0.7 SVR + 0.3 CNN) that smoothing variance and improves prediction stability.

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
│   ├── app.py                      ← Flask API Gateway
│   ├── cnn_inference.py            ← Satellite image analysis engine
│   ├── hybrid_model.py             ← Weighted fusion logic
│   ├── research_validation.py      ← High-fidelity metrics (MAE/RMSE)
│   ├── svr_model.py                ← SVR (Linear / RBF / Polynomial)
│   └── svr_tuning.py               ← GridSearchCV hyperparameter tuning
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
pip install -r requirements.txt

# 3. Running Research Metrics
python3 -m src.research_validation

# 4. Start the Prediction API
python3 src/app.py
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

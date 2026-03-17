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
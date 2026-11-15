# 🎵 Audio Virality Prediction

A lightweight, easy-to-explain Machine Learning project for predicting song virality from Spotify-style audio features using **Feature Selection**, **Principal Component Analysis (PCA)**, and **Classification Models**.

Built for academic coursework: minimal code, no unnecessary dependencies, no data leakage, and fully reproducible (`random_state=42`).

---

## 📌 Project Overview

- **Dataset**: `data/viral_songs.csv` (10,000 synthetic tracks, 2010–2025).
- **Target**: `viral` (Binary: `0 = Non-Viral` [63.95%], `1 = Viral` [36.05%]).
- **Leakage Prevention**: Post-release metrics (`streams`) and identifiers (`track_id`, `artist`, `track_name`, `release_year`) are dropped. All scalers, imputers, selectors, and PCA are fitted **strictly on the training set** (80/20 stratified split).
- **Techniques**:
  1. **Two-Stage Filter Feature Selection**: Correlation filter ($|r| > 0.85$) + Mutual Information (`mutual_info_classif`).
  2. **PCA (Dimensionality Reduction)**: Standardized PCA selecting the smallest component subset preserving $\ge 95\%$ variance (8 components, 95.40% retained).
  3. **Supervised Classification**: Logistic Regression vs. Random Forest, evaluated across 4 setups (with and without PCA).

---

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run each script in order:
```bash
python 01_explore_target.py      # Explores target distribution & class imbalance
python 02_feature_selection.py   # Selects top features via Correlation & Mutual Info
python 03_pca.py                 # Fits PCA, scree plot, loadings & variance analysis
python 04_models.py              # Evaluates 4 model setups with 5-fold CV & test set
```

All charts, metrics, and JSON metadata are automatically saved to the `results/` folder.

---

## 📊 Final Model Comparison

| Model | Setup | 5-Fold CV Accuracy | 5-Fold CV Macro F1 | Test Accuracy | Test Macro F1 | Test ROC-AUC | Train Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | `without_pca` | 0.6988 | 0.6931 | 70.60% | 0.6995 | **0.8048** | 0.029s |
| **Logistic Regression** | `with_pca` | 0.6979 | 0.6918 | **71.05%** | **0.7033** | 0.8046 | **0.018s** |
| **Random Forest** | `without_pca` | 0.6913 | 0.6760 | **71.25%** | 0.6965 | 0.7735 | 0.564s |
| **Random Forest** | `with_pca` | 0.6871 | 0.6713 | 68.90% | 0.6711 | 0.7632 | 0.589s |

> 🏆 **Best Model**: **Logistic Regression with PCA** (`Test Macro F1 = 0.7033`, `ROC-AUC = 0.8046`, `Accuracy = 71.05%`).
> PCA decorrelated linear features, improving both test generalization and training speed.

---

## 📁 Repository Structure

```
├── common.py               # Shared constants, load_data(), leak-free get_train_test()
├── 01_explore_target.py    # Target exploration and imbalance visualization
├── 02_feature_selection.py # Correlation and Mutual Information feature selection
├── 03_pca.py               # PCA fit, scree plot, loadings, and 2D scatter projection
├── 04_models.py            # Logistic Regression & Random Forest (with/without PCA)
├── requirements.txt        # Minimal Python dependencies
├── EXPLANATION.md          # Comprehensive viva & coursework theory explanation
├── data/
│   └── viral_songs.csv     # 10,000 tracks with audio features
└── results/                # Generated figures (.png), reports (.txt), and tables (.csv/.json)
```

For complete theory, PCA vs. LDA justification, metric definitions, and viva Q&A, refer to [`EXPLANATION.md`](EXPLANATION.md).

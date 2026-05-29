# Predicting Accident Severity on UK Road Traffic Data

Multi-stage data mining framework combining statistical feature selection (Cramér's V), ensemble classification (Random Forest, XGBoost with SMOTE), and Association Rule Mining (Apriori) on the 2024 UK STATS19 dataset (128K+ records) to predict collision severity and discover high-risk accident profiles. **CSC1144 – Data Analytics & Data Mining, DCU, Spring 2026.**

## Overview

Road accident severity is influenced by complex interactions between vehicle type, road environment, and driver behaviour — factors that traditional univariate analysis fails to capture. This project applies a three-phase data mining pipeline to the UK Department for Transport's STATS19 dataset to both **predict** accident severity (Fatal / Serious / Slight) and **discover** interpretable risk patterns for policy recommendations.

### Key Results

| Metric | Value |
|--------|-------|
| Best Model | Random Forest (Base) |
| Test Accuracy | 77.48% |
| Macro F1-Score | 0.6384 |
| Fatal Precision | 0.7742 |
| Highest Risk Profile Lift | 2.1× over baseline |

## Methodology

### Phase 1 — Statistical Feature Selection

- Merged three relational tables (Collision, Vehicle, Casualty) into a unified dataset of 128,266 records × 41 features
- Applied **bias-corrected Cramér's V** for categorical feature–target association and pairwise collinearity detection
- Removed redundant/low-variance features (coordinate duplicates, age bands, trunk road flag, near-zero variance columns)
- Engineered cyclical time features (hour sin/cos, month, weekend flag)

### Phase 2 — Predictive Modelling Under Class Imbalance

Evaluated **7 model configurations** across 3 algorithm families:

| Model | Accuracy | Macro-F1 | Fatal Precision |
|-------|----------|----------|-----------------|
| Logistic Regression | 0.5122 | 0.3712 | 0.0584 |
| **Random Forest (Base)** | **0.7748** | **0.6384** | **0.7742** |
| XGBoost (Base) | 0.6513 | 0.4964 | 0.1436 |
| RF v2 (FS + Depth) | 0.6701 | 0.5211 | 0.2180 |
| RF v2 + SMOTE | 0.7013 | 0.5077 | 0.1885 |
| XGBoost v2 (FS+SMOTE) | 0.7609 | 0.5285 | 0.3469 |
| XGBoost v3 (SMOTE) | 0.7845 | 0.5936 | 0.6245 |

- Addressed severe class imbalance (Fatal: 2%, Serious: 25%, Slight: 73%) via class weighting and **SMOTE**
- Hyperparameter tuning with **RandomizedSearchCV** (3-fold CV, optimised for Macro-F1)
- Key finding: SMOTE harms Random Forest but significantly boosts XGBoost's Fatal Precision (0.14 → 0.62)

### Phase 3 — Association Rule Mining

- Selected 16 interpretable features from XGBoost top-25 importance ranking
- Binarised features using Fatal/Serious rate threshold (27% baseline)
- **Apriori algorithm** (min_support=0.01) → filtered rules with Fatal/Serious as consequent
- Discovered dominant risk archetype (Lift ≈ 2.1): motorcycles >500cc and HGVs + risky manoeuvres + rural roads + speed ≥60mph

## Tech Stack

- **Language:** Python
- **ML / Data Mining:** scikit-learn, XGBoost, imbalanced-learn (SMOTE), mlxtend (Apriori)
- **Data Processing:** pandas, NumPy
- **Visualisation:** matplotlib, seaborn
- **Statistical Testing:** SciPy (chi-squared / Cramér's V)
- **Environment:** Google Colab

## Project Structure

```
├── data_mining_project.ipynb    # Full analysis notebook
├── report.pdf                   # Academic paper (IEEE format)
└── README.md
```

## My Contributions

This was a 4-person group project. I was responsible for:

- **EDA (Step 1):** All 6 exploratory visualisations (severity distribution, speed limit, light conditions, urban/rural, hour-of-day, correlation heatmap)
- **Data Preprocessing (Step 2):** Missing value imputation, categorical encoding (LabelEncoder), stratified train/test split, feature standardisation
- **Modelling (Step 3):** Implementation of all 3 baseline models (Logistic Regression, Random Forest, XGBoost) with class weighting and hyperparameter tuning
- **Model Improvement (Step 4):** Feature selection via XGBoost importance, all iterative model variants (RF v2, RF v2+SMOTE, XGBoost v2, XGBoost v3), 7-model comparison analysis, ROC curve analysis
- **Association Rule Mining (Step 5):** Full ARM pipeline — feature curation, binary encoding, Apriori execution, top-10 rule analysis, antecedent frequency analysis, vehicle type & manoeuvre risk breakdowns

## Data Source

[UK STATS19 Road Safety Open Data (2024)](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data) — UK Department for Transport, Open Government Licence.

## Acknowledgements

Group project for CSC1144 Data Analytics & Data Mining at Dublin City University. Team members: Megha Kanojia, Ananya Salil Warior, Shubin Li, Lamaan Shaikh.

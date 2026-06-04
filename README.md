# 🏦 Loan Default Predictor
### End-to-End Machine Learning Project | Give Me Some Credit Dataset

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-1%20EDA-blue)
![Day](https://img.shields.io/badge/Day-01%20of%2015-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Banks lose billions of dollars annually to loan defaults. This project builds a **machine learning model to predict whether a borrower will experience serious financial distress** (miss payments for 90+ days) in the next 2 years.

A well-calibrated model helps banks:
- Approve loans to creditworthy borrowers
- Flag high-risk applicants before issuing credit
- Reduce non-performing assets on the balance sheet

**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 borrower records from Kaggle.

---

## 🗂️ Project Structure

```
loan-default-predictor/
│
├── notebooks/
│   ├── Day_01_First_Look.ipynb          ✅ Done
│   ├── Day_02_Missing_Values.ipynb      🔒 Coming
│   ├── Day_03_Outlier_Detection.ipynb   🔒 Coming
│   ├── Day_04_Target_Analysis.ipynb     🔒 Coming
│   ├── Day_05_Correlation.ipynb         🔒 Coming
│   ├── Day_06_Imputation_Pipeline.ipynb 🔒 Coming
│   ├── Day_07_Feature_Engineering.ipynb 🔒 Coming
│   ├── Day_08_SMOTE.ipynb               🔒 Coming
│   ├── Day_09_Encoding_Scaling.ipynb    🔒 Coming
│   ├── Day_10_Train_Test_Split.ipynb    🔒 Coming
│   ├── Day_11_Logistic_Regression.ipynb 🔒 Coming
│   ├── Day_12_Random_Forest.ipynb       🔒 Coming
│   ├── Day_13_XGBoost.ipynb             🔒 Coming
│   ├── Day_14_SHAP_Explainability.ipynb 🔒 Coming
│   └── Day_15_Polish_and_Deploy.ipynb   🔒 Coming
│
├── outputs/
│   └── charts/                          # All saved visualizations
│
├── data/
│   └── DOWNLOAD_INSTRUCTIONS.md        # How to get the dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) |
| Rows | 150,000 borrowers |
| Features | 10 input features + 1 target |
| Target | `SeriousDlqin2yrs` (1 = defaulted, 0 = did not) |
| Class Balance | ~93% No Default / ~7% Default |

### Feature Dictionary

| Column | Type | Description |
|---|---|---|
| `SeriousDlqin2yrs` | Target | 1 = defaulted within 2 years |
| `RevolvingUtilizationOfUnsecuredLines` | Float | % of credit limit in use |
| `age` | Int | Borrower's age |
| `NumberOfTime30-59DaysPastDueNotWorse` | Int | Times 30-59 days late |
| `DebtRatio` | Float | Monthly debt / monthly income |
| `MonthlyIncome` | Float | Monthly income in USD |
| `NumberOfOpenCreditLinesAndLoans` | Int | Open credit lines + loans |
| `NumberOfTimes90DaysLate` | Int | Times 90+ days late |
| `NumberRealEstateLoansOrLines` | Int | Mortgage + real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Int | Times 60-89 days late |
| `NumberOfDependents` | Float | Number of family dependents |

---

## 🗺️ 15-Day Roadmap

### Phase 1 — EDA (Days 1–5)
| Day | Topic | Status |
|---|---|---|
| Day 01 | First Look — Load, shape, dtypes, missing scan, target distribution | ✅ **Done** |
| Day 02 | Missing Values — missingno, imputation strategy, flag columns | 🔒 Upcoming |
| Day 03 | Outlier Detection — IQR method, box plots, cap vs remove decisions | 🔒 Upcoming |
| Day 04 | Target Variable — Default rate by age/income, distribution splits | 🔒 Upcoming |
| Day 05 | Correlation — Heatmap, multicollinearity, pairplot, EDA summary | 🔒 Upcoming |

### Phase 2 — Feature Engineering (Days 6–10)
| Day | Topic | Status |
|---|---|---|
| Day 06 | Imputation Pipeline (sklearn Pipeline) | 🔒 Upcoming |
| Day 07 | New Features — debt-to-income ratio, late payment streaks | 🔒 Upcoming |
| Day 08 | Class Imbalance — SMOTE on training data only | 🔒 Upcoming |
| Day 09 | Encoding + Scaling | 🔒 Upcoming |
| Day 10 | Stratified Train/Test Split + Validation Strategy | 🔒 Upcoming |

### Phase 3 — Modeling & Explainability (Days 11–15)
| Day | Topic | Status |
|---|---|---|
| Day 11 | Logistic Regression — Baseline model | 🔒 Upcoming |
| Day 12 | Random Forest — Feature importance | 🔒 Upcoming |
| Day 13 | XGBoost — Production-grade model, AUC-ROC comparison | 🔒 Upcoming |
| Day 14 | SHAP Values — Why did the model flag this borrower? | 🔒 Upcoming |
| Day 15 | Polish + Streamlit App + GitHub cleanup | 🔒 Upcoming |

---

## 📅 Day 01 — First Look at the Data

> **Notebook:** [`notebooks/Day_01_First_Look.ipynb`](notebooks/Day_01_First_Look.ipynb)

### What Was Done
- Loaded `cs-training.csv` (150,000 rows × 11 columns)
- Inspected data types using `.info()` — all numeric, no text columns
- Generated statistical summary using `.describe()` — identified skewness in income and debt ratio
- Scanned for missing values with `.isnull().sum()`
- Analyzed target variable distribution
- Plotted distributions for all features

### Key Findings

**Missing Data**
- `MonthlyIncome`: ~20% missing — significant, will require careful imputation
- `NumberOfDependents`: ~2.6% missing — likely 0-dependent borrowers who left it blank

**Target Variable**
- Default rate: ~6.7% → dataset is **highly imbalanced** (~14:1 ratio)
- ⚠️ Accuracy is a misleading metric here — a model predicting "no default" for everyone scores 93.3% accuracy but detects zero actual defaults
- **Will use AUC-ROC as primary metric throughout this project**

**Suspicious Values (flagged for Day 3)**
- `age` has a minimum of 0 — impossible, will remove
- `DebtRatio` max is extremely high — likely data entry errors
- Late payment columns contain values of 96/98 — known Kaggle artifact, will cap

### Charts Generated
| Chart | Description |
|---|---|
| `missing_values.png` | Bar chart of missing % per column |
| `target_distribution.png` | Class distribution — bar + pie |
| `feature_distributions.png` | All feature histograms (capped at 99th percentile) |

---
## ✅ Day 02 — Missing Values Deep Dive

> **Notebook:** [`notebooks/Day_02_Missing_Values.ipynb`](notebooks/Day_02_Missing_Values.ipynb)

### What Was Done
- Visualized missingness patterns using `missingno` (matrix, bar, heatmap)
- Classified missingness type per column (MCAR / MAR / MNAR)
- **Critical analysis:** Checked if missing income correlates with higher default rate
- Built imputation decision table with documented reasoning
- Applied stratified median imputation for `MonthlyIncome` (by age group)
- Filled `NumberOfDependents` missing with 0
- Added `MonthlyIncome_was_missing` binary flag column **before** imputing
- Verified imputation with before/after distribution comparison
- Saved cleaned dataset for Day 03

### Key Findings

| Finding | Detail |
|---|---|
| Missingness type: MonthlyIncome | MAR/MNAR — borrowers hiding income tend to be riskier |
| Missingness type: NumberOfDependents | MAR — blank likely means 0 dependents |
| Missing income → default rate | Higher default rate in missing income group → missingness IS informative |
| Flag column added | `MonthlyIncome_was_missing` preserves the signal before imputing |
| Imputation strategy: MonthlyIncome | Stratified median by age group (not global median) |
| Imputation strategy: NumberOfDependents | Fill with 0 |

### The 3 Types of Missingness (Interview Concept)

| Type | Meaning | Example in this dataset |
|---|---|---|
| **MCAR** — Missing Completely At Random | Random, no pattern | — |
| **MAR** — Missing At Random | Depends on other columns | Older people less likely to report income |
| **MNAR** — Missing Not At Random | Depends on the missing value itself | High-risk borrowers hiding income |

> ⚠️ MNAR is the most dangerous — imputing without a flag gives risky borrowers a fake "safe" income value. Always add a flag column first.

### Charts Generated
| Chart | Description |
|---|---|
| `msno_matrix.png` | White lines show which rows are missing — look for alignment |
| `msno_bar.png` | Column completeness bar chart |
| `msno_heatmap.png` | Do columns go missing together? |
| `missing_income_vs_default.png` | Default rate: income-provided vs income-missing |
| `income_imputation_comparison.png` | Distribution before vs after imputation |

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` + `seaborn` | Visualization |
| `missingno` | Missing value visualization |
| `scikit-learn` | Preprocessing, models, metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `xgboost` | Gradient boosted trees |
| `shap` | Model explainability |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-default-predictor.git
cd loan-default-predictor

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# See data/DOWNLOAD_INSTRUCTIONS.md
# Place cs-training.csv inside the /data/ folder

# 5. Run notebooks in order
jupyter notebook
# Open notebooks/ → start from Day_01_First_Look.ipynb
```

---

## 📈 Results (Updated as project progresses)

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

*Results will be filled in during Days 11–13.*

---

## 💡 Key Learnings (Updated daily)

- **Day 01:** Class imbalance of 14:1 means accuracy is a useless metric for this problem. AUC-ROC is the right choice.

---

## 🧠 Interview Talking Points

> These grow as the project progresses.

- *"The dataset had a 14:1 class imbalance — I chose AUC-ROC over accuracy because a model predicting 'no default' for everyone still gets 93% accuracy but is completely useless to a bank."*

---

## 📄 License

MIT License — free to use, modify, and share.

---

*Built as part of a structured 15-day ML learning sprint — one notebook per day, one concept at a time.*

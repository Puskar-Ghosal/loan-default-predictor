# 🏦 Loan Default Predictor
### End-to-End Machine Learning Project | Give Me Some Credit Dataset

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-2%20Feature%20Engineering-purple)
![Day](https://img.shields.io/badge/Day-06%20of%2015-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

Banks lose billions of dollars annually to loan defaults. This project builds a **machine learning model to predict whether a borrower will experience serious financial distress** (miss payments for 90+ days) in the next 2 years.

**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 borrower records from Kaggle.

---

## 🗂️ Project Structure

```
loan-default-predictor/
│
├── notebooks/
│   ├── Day_01_First_Look.ipynb           ✅ Done
│   ├── Day_02_Missing_Values.ipynb       ✅ Done
│   ├── Day_03_Outlier_Detection.ipynb    ✅ Done
│   ├── Day_04_Target_Analysis.ipynb      ✅ Done
│   ├── Day_05_Correlation.ipynb          ✅ Done
│   ├── Day_06_Imputation_Pipeline.ipynb  ✅ Done
│   ├── Day_07_Feature_Engineering.ipynb  🔒 Coming
│   ├── Day_08_SMOTE.ipynb                🔒 Coming
│   ├── Day_09_Encoding_Scaling.ipynb     🔒 Coming
│   ├── Day_10_Train_Test_Split.ipynb     🔒 Coming
│   ├── Day_11_Logistic_Regression.ipynb  🔒 Coming
│   ├── Day_12_Random_Forest.ipynb        🔒 Coming
│   ├── Day_13_XGBoost.ipynb              🔒 Coming
│   ├── Day_14_SHAP_Explainability.ipynb  🔒 Coming
│   └── Day_15_Polish_and_Deploy.ipynb    🔒 Coming
│
├── outputs/
│   └── charts/
│
├── data/
│   └── DOWNLOAD_INSTRUCTIONS.md
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

### Phase 1 — EDA (Days 1–5) ✅ Complete

| Day | Topic | Status |
|---|---|---|
| Day 01 | First Look — Load, shape, dtypes, missing scan, target distribution | ✅ Done |
| Day 02 | Missing Values — missingno, imputation strategy, flag columns | ✅ Done |
| Day 03 | Outlier Detection — IQR method, box plots, cap vs remove | ✅ Done |
| Day 04 | Target Variable — Default rate by age/income, compare two groups | ✅ Done |
| Day 05 | Correlation — Heatmap, feature vs target, multicollinearity | ✅ Done |

### Phase 2 — Feature Engineering (Days 6–10)

| Day | Topic | Status |
|---|---|---|
| Day 06 | sklearn Pipeline — imputer + scaler in one reusable object | ✅ **Done** |
| Day 07 | New Features — debt-to-income ratio, total late payments, ever late flag | 🔒 Upcoming |
| Day 08 | SMOTE — create synthetic minority samples to fix class imbalance | 🔒 Upcoming |
| Day 09 | Encoding + Scaling — scale train and test correctly | 🔒 Upcoming |
| Day 10 | Stratified Split + correct order of all operations | 🔒 Upcoming |

### Phase 3 — Modeling & Explainability (Days 11–15)

| Day | Topic | Status |
|---|---|---|
| Day 11 | Logistic Regression — Baseline model | 🔒 Upcoming |
| Day 12 | Random Forest — Feature importance | 🔒 Upcoming |
| Day 13 | XGBoost — Production-grade model, AUC-ROC comparison | 🔒 Upcoming |
| Day 14 | SHAP Values — Why did the model flag this borrower? | 🔒 Upcoming |
| Day 15 | Polish + Streamlit App + GitHub cleanup | 🔒 Upcoming |

---

## ✅ Phase 1 Summary — EDA (Days 1–5)

| Topic | What We Found |
|---|---|
| Dataset size | ~150,000 rows, 11 columns, all numeric |
| Missing data | MonthlyIncome (~20%), NumberOfDependents (~2.6%) |
| Outliers | Impossible ages removed, extreme values capped |
| Class imbalance | 14:1 ratio — will use AUC-ROC and SMOTE |
| Strongest features | Late payment history and credit utilization |
| Multicollinearity | 3 late payment columns overlap — to be handled in Day 07 |

---

## ✅ Day 06 — sklearn Pipeline

> **Notebook:** [`notebooks/Day_06_Imputation_Pipeline.ipynb`](notebooks/Day_06_Imputation_Pipeline.ipynb)

### What Was Done
- Learned what a sklearn Pipeline is and why it is useful
- Added the `MonthlyIncome_was_missing` flag column before the pipeline
- Removed impossible age rows before the pipeline
- Capped all outlier columns before the pipeline
- Built a 2-step pipeline: SimpleImputer (fills missing) + StandardScaler (scales values)
- Used `fit_transform()` on data and checked output
- Verified all missing values are gone after pipeline
- Compared column values before and after scaling

### Key Findings

| Finding | Detail |
|---|---|
| Pipeline steps | 1. Imputer fills remaining nulls with median  2. Scaler centers values around 0 |
| Age before scaling | Values between 18 and 100 |
| Age after scaling | Values roughly between -3 and +3 |
| Why scaling matters | Without it, income (0–20000) dominates age (18–100) in some models |

### What is a Pipeline?

Instead of doing this manually every time:
```python
imputer.fit(X_train)
X_train = imputer.transform(X_train)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# then repeat all steps for test data...
```

A pipeline lets you do it in one line:
```python
pipeline = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)   # same steps applied automatically
```

### What is StandardScaler?
```
For each column:
  scaled_value = (original_value - mean) / standard_deviation

Result: every column has mean = 0 and std = 1
All columns are now on the same scale
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `seaborn` | Heatmaps and pairplots |
| `missingno` | Missing value visualization |
| `scikit-learn` | Preprocessing, pipeline, models, metrics |
| `imbalanced-learn` | SMOTE for class imbalance |
| `xgboost` | Gradient boosted trees |
| `shap` | Model explainability |

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/loan-default-predictor.git
cd loan-default-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset — see data/DOWNLOAD_INSTRUCTIONS.md

# 5. Run notebooks in order
jupyter notebook
```

---

## 📈 Results (Updated as project progresses)

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

*Will be filled in Days 11–13*

---

## 💡 Key Learnings (Updated daily)

- **Day 01:** Always look at your data first. `.info()`, `.describe()`, `.isnull().sum()` on every new dataset.
- **Day 02:** Never fill missing values blindly. Check if the missingness itself is informative first.
- **Day 03:** An outlier is not always an error. Cap real outliers, remove impossible ones.
- **Day 04:** 93% accuracy on this dataset means nothing — a model saying "no default" always gets that. Use AUC-ROC.
- **Day 05:** High correlation between two INPUT features is a problem. High correlation between input and TARGET is good.
- **Day 06:** A Pipeline keeps all preprocessing steps in one object. fit on train, transform on test — never the other way.

---

## 🎯 Interview Talking Points (Updated daily)

- *"The dataset had a 14:1 class imbalance — I used AUC-ROC instead of accuracy."*
- *"I added a binary flag for missing income before imputing — missing income is itself a risk signal."*
- *"I used Winsorization instead of row deletion for outliers — keeps real data, tames extreme values."*
- *"From feature analysis, NumberOfTimes90DaysLate had the biggest separation between the two groups."*
- *"I found multicollinearity between the three late payment columns and combined them in feature engineering."*
- *"I used a sklearn Pipeline so that all preprocessing steps are applied consistently to both train and test data. This prevents data leakage."*

---

## 📄 License

MIT License — free to use, modify, and share.

---

*Built as part of a structured 15-day ML learning sprint — one notebook per day, one concept at a time.*

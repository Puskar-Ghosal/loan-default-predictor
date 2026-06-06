# 🏦 Loan Default Predictor
### End-to-End Machine Learning Project | Give Me Some Credit Dataset

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-1%20EDA-blue)
![Day](https://img.shields.io/badge/Day-03%20of%2015-orange)
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
│   ├── Day_01_First_Look.ipynb           ✅ Done
│   ├── Day_02_Missing_Values.ipynb       ✅ Done
│   ├── Day_03_Outlier_Detection.ipynb    ✅ Done
│   ├── Day_04_Target_Analysis.ipynb      🔒 Coming
│   ├── Day_05_Correlation.ipynb          🔒 Coming
│   ├── Day_06_Imputation_Pipeline.ipynb  🔒 Coming
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
│   └── charts/                           # All saved visualizations
│
├── data/
│   └── DOWNLOAD_INSTRUCTIONS.md         # How to get the dataset
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
| Day 02 | Missing Values — missingno, imputation strategy, flag columns | ✅ **Done** |
| Day 03 | Outlier Detection — IQR method, box plots, cap vs remove decisions | ✅ **Done** |
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

## ✅ Day 01 — First Look at the Data

> **Notebook:** [`notebooks/Day_01_First_Look.ipynb`](notebooks/Day_01_First_Look.ipynb)

### What Was Done
- Loaded `cs-training.csv` — 150,000 rows × 11 columns
- Inspected data types with `.info()` — all numeric, no text columns
- Generated statistical summary with `.describe()` — identified skewness in income and debt ratio
- Scanned for missing values with `.isnull().sum()`
- Analyzed target variable class distribution
- Plotted histograms for all 10 features

### Key Findings

| Finding | Detail |
|---|---|
| Missing: MonthlyIncome | ~20% missing — needs careful imputation |
| Missing: NumberOfDependents | ~2.6% missing — likely means 0 dependents |
| Default rate | ~6.7% → 14:1 class imbalance |
| Metric decision | AUC-ROC chosen over accuracy (imbalance problem) |
| Suspicious: age = 0 | Impossible — flagged for Day 03 |
| Suspicious: DebtRatio max | Extremely high — likely data entry error |

### Charts Generated
| Chart | Description |
|---|---|
| `missing_values.png` | Bar chart of missing % per column |
| `target_distribution.png` | Class distribution — bar + pie |
| `feature_distributions.png` | All feature histograms (capped at 99th pct) |

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

### Key Findings

| Finding | Detail |
|---|---|
| Missingness type: MonthlyIncome | MAR/MNAR — borrowers hiding income tend to be riskier |
| Missingness type: NumberOfDependents | MAR — blank likely means 0 dependents |
| Missing income → default rate | Higher default rate in missing income group → missingness IS informative |
| Flag column added | `MonthlyIncome_was_missing` preserves signal before imputing |
| Imputation: MonthlyIncome | Stratified median by age group (not global median) |
| Imputation: NumberOfDependents | Fill with 0 |

### The 3 Types of Missingness (Interview Concept)

| Type | Meaning | Example in this dataset |
|---|---|---|
| **MCAR** — Missing Completely At Random | Random, no pattern | — |
| **MAR** — Missing At Random | Depends on other columns | Older people less likely to report income |
| **MNAR** — Missing Not At Random | Depends on the missing value itself | High-risk borrowers hiding income |

### Charts Generated
| Chart | Description |
|---|---|
| `msno_matrix.png` | White lines show missing rows — look for alignment |
| `msno_bar.png` | Column completeness bar chart |
| `msno_heatmap.png` | Do columns go missing together? |
| `missing_income_vs_default.png` | Default rate: income-provided vs income-missing |
| `income_imputation_comparison.png` | Distribution before vs after imputation |

---

## ✅ Day 03 — Outlier Detection

> **Notebook:** [`notebooks/Day_03_Outlier_Detection.ipynb`](notebooks/Day_03_Outlier_Detection.ipynb)

### What Was Done
- Plotted box plots for every numeric feature to visually identify outliers
- Applied the **IQR method** to quantify outlier count and % per column
- Ran **domain-specific validity checks** (impossible ages, extreme utilization, known Kaggle artifacts)
- Built an outlier decision table classifying each as ERROR vs REAL
- Implemented treatment: row removal for impossible values, capping (Winsorization) for extreme-but-real values
- Verified treatment with before/after box plot comparisons
- Saved cleaned dataset for Day 04

### Key Findings

| Column | Issue Found | Treatment Applied |
|---|---|---|
| `age` | age = 0 and age < 18 found — legally impossible | Removed rows |
| `RevolvingUtilizationOfUnsecuredLines` | Max > 1000% — extreme beyond realistic over-limit | Capped at 1.5 (150%) |
| `DebtRatio` | Max in millions — clear data entry errors | Capped at 99th percentile |
| `MonthlyIncome` | Extreme high-end values skew the distribution | Capped at 99th percentile |
| Late payment columns | Values of 96/98 appear frequently — known Kaggle artifact | Capped at 10 |

### Two Types of Outliers (Interview Concept)

| Type | Description | Treatment |
|---|---|---|
| **Error outlier** | Impossible in reality — age = 0, income = -500 | Remove the row |
| **Real outlier** | Extreme but valid — income = $500k/month | Cap (Winsorize) — never delete real data |

### What is Winsorization?
Instead of **deleting** outlier rows (which loses real borrower data), we **clip** values to a ceiling:
- A DebtRatio of 999,999 → clipped to the 99th percentile value
- The row stays in the dataset, the extreme value is tamed
- Model training becomes more stable without losing data volume

### IQR Method Formula
```
Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1

Lower fence = Q1 - (1.5 × IQR)
Upper fence = Q3 + (1.5 × IQR)

Values outside these fences = outliers
```

### Charts Generated
| Chart | Description |
|---|---|
| `boxplots_all_features.png` | Box plots for every feature — dots beyond whiskers = outliers |
| `outlier_before_after.png` | Before (red) vs after (green) for 3 most affected columns |

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

- **Day 01:** Class imbalance of 14:1 means accuracy is a useless metric. AUC-ROC is the right choice for this problem.
- **Day 02:** Never impute blindly. Missing values have reasons — understanding WHY they're missing changes your strategy completely. Always add a flag column before imputing MNAR data.
- **Day 03:** An outlier is not automatically an error. A person earning $500k/month is a real outlier — deleting them loses real data. Cap them instead. But age = 0 is a data error — remove those rows. Always distinguish before acting.

---

## 🎯 Interview Talking Points (Updated daily)

- *"The dataset had a 14:1 class imbalance — I chose AUC-ROC over accuracy because a model predicting 'no default' for everyone still gets 93% accuracy but is completely useless to a bank."*
- *"I found that borrowers with missing income had a higher default rate — so I added a binary flag column before imputing with median. This way the model can learn that 'income not provided' is itself a risk signal."*
- *"I used stratified median imputation for MonthlyIncome — stratified by age group — because a 25-year-old's typical income is very different from a 55-year-old's. Global median would have introduced bias."*
- *"For outliers I used Winsorization instead of deletion. Removing rows with extreme income values would erase real borrowers from the dataset. Capping at the 99th percentile tames the extreme values while keeping every row."*
- *"The late payment columns had values of 96 and 98 appearing suspiciously often — this is a known artifact in this Kaggle dataset. I capped those at 10 to prevent the model from treating them as meaningful high counts."*

---

## 📄 License

MIT License — free to use, modify, and share.

---

*Built as part of a structured 15-day ML learning sprint — one notebook per day, one concept at a time.*

# 🏦 Loan Default Predictor
### End-to-End Machine Learning Project | Give Me Some Credit Dataset

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-1%20EDA-blue)
![Day](https://img.shields.io/badge/Day-05%20of%2015-orange)
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
│   ├── Day_04_Target_Analysis.ipynb      ✅ Done
│   ├── Day_05_Correlation.ipynb          ✅ Done
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
| Day 04 | Target Variable — Default rate by age/income, compare two groups | ✅ **Done** |
| Day 05 | Correlation — Heatmap, feature vs target, multicollinearity, pairplot | ✅ **Done** |

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
- Checked data types with `.info()` — all numeric, no text columns
- Looked at basic statistics with `.describe()`
- Counted missing values with `.isnull().sum()`
- Plotted target column distribution
- Plotted histogram for every feature

### Key Findings

| Finding | Detail |
|---|---|
| Missing: MonthlyIncome | ~20% missing |
| Missing: NumberOfDependents | ~2.6% missing |
| Default rate | ~6.7% — very imbalanced dataset |
| age min = 0 | Impossible — flagged for Day 03 |
| DebtRatio max is huge | Likely a data entry error |

---

## ✅ Day 02 — Missing Values

> **Notebook:** [`notebooks/Day_02_Missing_Values.ipynb`](notebooks/Day_02_Missing_Values.ipynb)

### What Was Done
- Used `missingno` library to visualize missing patterns
- Checked if missing income people default more (answered: yes)
- Added `income_is_missing` flag column before filling values
- Filled `MonthlyIncome` with median
- Filled `NumberOfDependents` with 0
- Compared income distribution before and after filling

### Key Findings

| Finding | Detail |
|---|---|
| Missing income → default rate | Higher than people who provided income |
| Why median not mean for income | Income data is skewed — median is more reliable |
| Why fill dependents with 0 | Missing likely means no dependents |
| `income_is_missing` column | Keeps the information that income was hidden |

---

## ✅ Day 03 — Outlier Detection

> **Notebook:** [`notebooks/Day_03_Outlier_Detection.ipynb`](notebooks/Day_03_Outlier_Detection.ipynb)

### What Was Done
- Plotted box plots for each column to see outliers visually
- Used IQR method to calculate exact outlier boundaries
- Checked domain-specific impossible values (age < 18, etc.)
- Removed rows where age < 18
- Capped extreme values instead of deleting rows (Winsorization)

### Key Findings

| Column | Problem | Fix Applied |
|---|---|---|
| `age` | age < 18 — cannot legally have a loan | Removed those rows |
| `RevolvingUtilizationOfUnsecuredLines` | Max way above 100% | Capped at 1.5 |
| `DebtRatio` | Extremely high values | Capped at 99th percentile |
| `MonthlyIncome` | Very extreme high values | Capped at 99th percentile |
| Late payment columns | Values 96/98 — known dataset error | Capped at 10 |

### IQR Formula
```
Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1

Lower boundary = Q1 - (1.5 × IQR)
Upper boundary = Q3 + (1.5 × IQR)
```

---

## ✅ Day 04 — Understanding the Target Column

> **Notebook:** [`notebooks/Day_04_Target_Analysis.ipynb`](notebooks/Day_04_Target_Analysis.ipynb)

### What Was Done
- Split data into defaulters and non-defaulters
- Compared average age, income, utilization between both groups
- Created age groups and checked default rate in each
- Created income brackets and checked default rate in each
- Compared all features at once using a mean comparison table
- Proved why accuracy is a bad metric using real numbers

### Key Findings

| Question | Answer |
|---|---|
| Riskiest age group | Younger borrowers (18–30) default the most |
| Income vs default | Lower income = higher default rate |
| Strongest signal | `NumberOfTimes90DaysLate` — biggest gap between groups |
| Dummy model accuracy | 93.3% accuracy but catches 0 defaulters |
| Right metric | AUC-ROC — not accuracy |

---

## ✅ Day 05 — Correlation Between Features

> **Notebook:** [`notebooks/Day_05_Correlation.ipynb`](notebooks/Day_05_Correlation.ipynb)

### What Was Done
- Calculated correlation between every feature using `.corr()`
- Plotted a heatmap to see all correlations at once
- Pulled out just the correlation of each feature with our target
- Ranked features from most to least correlated with default
- Found feature pairs that are highly correlated with each other (multicollinearity)
- Made a pairplot of the top 4 features colored by default vs no default
- Made a scatter plot of utilization vs late payments to see if they separate the two groups

### Key Findings

| Finding | Detail |
|---|---|
| Most correlated with default | `NumberOfTimes90DaysLate` — strongest signal |
| Second most correlated | `NumberOfTime30-59DaysPastDueNotWorse` |
| Multicollinearity found | The three late payment columns are correlated with each other |
| Why late columns are correlated | If someone is 90 days late, they were probably 30 days late first |
| What to do about it | Will handle in Feature Engineering — Day 07 |

### What is Correlation?

```
Correlation value is between -1 and +1

+1  = both go up together (e.g. height and weight)
 0  = no relationship at all
-1  = one goes up, the other goes down
```

### What is Multicollinearity?

When two input features are highly correlated with each other, they are saying the same thing.
Keeping both can confuse some models (especially Logistic Regression).
The fix is to drop one of the pair — we keep whichever one is more correlated with the target.

### Charts Made Today
| Chart | What it shows |
|---|---|
| Correlation heatmap | All features vs all features — red = positive, blue = negative |
| Feature vs target bar chart | Which features are most related to default |
| Pairplot (top 4 features) | Visual separation between defaulters and non-defaulters |
| Scatter plot | Utilization vs late payments — colored by default |

---

## 🏁 Phase 1 Complete — EDA Done

After 5 days of EDA, here is what we know about this dataset:

| Topic | Summary |
|---|---|
| Dataset size | ~150,000 rows, 11 columns |
| Missing data | Fixed — income filled with median, dependents filled with 0 |
| Outliers | Fixed — impossible ages removed, extreme values capped |
| Class imbalance | 14:1 ratio — will use AUC-ROC and SMOTE |
| Strongest features | Late payment history and credit utilization |
| Weak features | NumberOfOpenCreditLines, NumberRealEstateLoans |
| Multicollinearity | 3 late payment columns overlap — will simplify in Day 07 |

**Phase 2 starts tomorrow — Feature Engineering.**

---

## 🧰 Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization |
| `seaborn` | Heatmaps and pairplots |
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

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# See data/DOWNLOAD_INSTRUCTIONS.md
# Place cs-training.csv inside the /data/ folder

# 5. Open Jupyter and run notebooks in order
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

- **Day 01:** Always look at your data first before writing any code. `.info()`, `.describe()`, `.isnull().sum()` are the three commands you run on every new dataset.
- **Day 02:** Never fill missing values blindly. First check if the missingness itself tells you something. In this dataset, people with missing income default more — so we saved that as a separate column before filling.
- **Day 03:** An outlier is not always an error. A person earning $500k/month is extreme but real. Deleting them loses real data. Capping keeps the row but controls the extreme number.
- **Day 04:** A model predicting "no default" for everyone gets 93% accuracy but is completely useless. Always think about what the metric actually means for the business.
- **Day 05:** High correlation between two INPUT features is a problem — they say the same thing. High correlation between an input feature and the TARGET is what we want — it means that feature is useful for prediction.

---

## 🎯 Interview Talking Points (Updated daily)

- *"The dataset had a 14:1 class imbalance — I chose AUC-ROC over accuracy because a model predicting no default for everyone gets 93% accuracy but catches zero actual defaulters."*
- *"I found that borrowers with missing income had a higher default rate. So I added a binary flag column before filling the missing values — the model can then learn that hiding income is itself a risk signal."*
- *"For outliers I used capping instead of deletion. Removing rows with extreme values loses real borrower data. Capping at the 99th percentile tames the extreme values while keeping every row."*
- *"From the feature comparison, NumberOfTimes90DaysLate showed the biggest average difference between defaulters and non-defaulters — which later confirmed its high feature importance in the XGBoost model."*
- *"I found multicollinearity between the three late payment columns — 30-day, 60-day, and 90-day late. This makes business sense since someone who is 90 days late was almost certainly 30 days late first. I handled this during feature engineering."*

---

## 📄 License

MIT License — free to use, modify, and share.

---

*Built as part of a structured 15-day ML learning sprint — one notebook per day, one concept at a time.*

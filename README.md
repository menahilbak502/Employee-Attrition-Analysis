# Employee Attrition Analysis
### HR Analytics

A full machine learning pipeline to predict employee attrition, identify its key drivers, quantify its business cost, and develop role-specific retention strategies — built using methods from BA305.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methods Used](#methods-used)
- [Key Results](#key-results)
- [Business Cost Analysis](#business-cost-analysis)
- [Role-Specific Findings](#role-specific-findings)
- [Action Plan](#action-plan)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

---

## Project Overview

This project answers three core business questions using machine learning:

1. **Who is likely to leave?** — Build and compare 6 classification models to predict attrition
2. **What are the top drivers?** — Identify actionable levers (overtime, pay, commute, tenure, role) via a risk dashboard
3. **What does it cost?** — Evaluate models not just on accuracy but on dollar value saved, and identify the best model for business impact

The analysis is structured across 5 Google Colab code blocks, each building on the last.

---

## Dataset

**Source:** [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/thedevastator/employee-attrition-and-factors)

| Property | Value |
|---|---|
| Rows | 1,470 employees |
| Features | 35 original, 44 after encoding |
| Target Variable | Attrition (Yes/No) |
| Attrition Rate | 16.1% (237 leavers) |
| Class Balance | ~5 stayers per 1 leaver |

Key features include Age, MonthlyIncome, OverTime, JobRole, DistanceFromHome, YearsAtCompany, JobSatisfaction, TotalWorkingYears, and more.

---

## Project Structure

```
├── README.md
├── HR_Analytics_csv.csv           # Source dataset
├── Personal_Project.ipynb         # Main Google Colab notebook (all 5 blocks)
└── attrition_report.docx          # Full results report
```

The notebook is organized into 5 sequential blocks — each block must be run in order:

| Block | Description |
|---|---|
| Block 1 | Setup, EDA, preprocessing, train/test split |
| Block 2 | 6 classification models + comparison table + ROC curves |
| Block 3 | Attrition risk dashboard — feature importance, odds ratios, actionable levers |
| Block 4 | Cost of attrition model — business value evaluation by model |
| Block 5 | Role-specific models — separate Random Forests per job role group |

---

## Methods Used

All methods are drawn from BA305 course content:

- **Logistic Regression** — primary model; odds ratios for interpretability
- **Decision Tree** (max depth=5) — human-readable rules; overfitting controlled via pruning
- **Random Forest** (500 trees) — ensemble method for feature importance
- **Neural Network** (16-8 hidden layers, ReLU, Adam) — non-linear pattern detection
- **K-Nearest Neighbors** (k=11) — distance-based classification
- **Naive Bayes** — probabilistic classifier; highest recall

**Evaluation metrics used:**
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Business cost matrix (custom dollar-value evaluation)
- Threshold optimization (finding optimal classification cutoff)

**Preprocessing:**
- Stratified train/test split (70/30) to preserve class balance
- StandardScaler normalization for LR, KNN, Neural Net
- One-hot encoding for all categorical features
- Dropped constant/ID columns: EmployeeCount, EmployeeNumber, Over18, StandardHours

---

## Key Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.725 | 0.408 | 0.523 | 0.820 |
| Decision Tree (depth=5) | 0.832 | 0.457 | 0.225 | 0.302 | 0.693 |
| Random Forest | 0.830 | 0.409 | 0.127 | 0.194 | 0.774 |
| Neural Network (16-8) | 0.832 | 0.477 | 0.437 | 0.456 | 0.737 |
| KNN (k=11) | 0.841 | 0.556 | 0.070 | 0.125 | 0.686 |
| Naive Bayes | 0.635 | 0.266 | 0.718 | 0.388 | 0.709 |

> **Note:** Accuracy is misleading with a 16% attrition rate. A model predicting "Stay" for everyone would score 83.9% accuracy while catching zero leavers. ROC-AUC and Recall are the primary metrics of interest.

### Top Attrition Drivers (Random Forest Feature Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | MonthlyIncome | 0.079 |
| 2 | Age | 0.063 |
| 3 | TotalWorkingYears | 0.060 |
| 4 | DailyRate | 0.052 |
| 5 | HourlyRate | 0.049 |
| 6 | MonthlyRate | 0.046 |
| 7 | YearsAtCompany | 0.043 |
| 8 | DistanceFromHome | 0.042 |
| 9 | YearsWithCurrManager | 0.038 |
| 10 | OverTime_Yes | 0.037 |

Compensation variables alone account for over 22% of total model explanatory power.

### Key Dashboard Findings

| Lever | Finding |
|---|---|
| OverTime | 30.6% attrition with OT vs 10.4% without — nearly 3x |
| Income Q1 vs Q4 | 27-28% attrition in bottom quartile vs 8-10% in top quartile |
| Tenure under 1 year | 35% attrition — the early tenure cliff |
| Sales Representatives | ~40% attrition — highest of any role |
| Commute 21-30 miles | Elevated attrition vs short-commute employees |

### Decision Tree Rules

The top-level rules from the depth-5 tree reveal the clearest human-readable logic:

```
TotalWorkingYears <= 1.5 years
├── Age <= 28.5 → HIGH RISK (class: 1)
└── Age > 28.5  → Lower risk (class: 0)

TotalWorkingYears > 1.5 years
├── OverTime = No → Lower risk
└── OverTime = Yes
    ├── MonthlyIncome <= 2,476 → HIGH RISK (class: 1)
    └── MonthlyIncome > 2,476 → Context-dependent
```

---

## Business Cost Analysis

### Assumptions

| Parameter | Value |
|---|---|
| Average Annual Salary | 78,035 |
| Replacement Cost per Employee | 78,035 (100% of annual salary) |
| Intervention Cost per Employee | 3,902 (5% of annual salary) |
| Baseline Cost (no model) | 5,540,497 |

### Model Business Value (Test Set)

| Model | TP | FP | FN | Net Cost | Savings vs Baseline |
|---|---|---|---|---|---|
| KNN (k=11) | 5 | 4 | 66 | 4,795,262 | 745,235 |
| Random Forest | 9 | 13 | 62 | 4,221,703 | 1,318,794 |
| Decision Tree | 16 | 19 | 55 | 3,179,933 | 2,360,564 |
| Logistic Regression | 29 | 11 | 42 | 1,170,528 | 4,369,969 |
| Neural Network | 31 | 34 | 40 | 955,931 | 4,584,566 |
| Naive Bayes | 51 | 141 | 20 | -1,669,953 | 7,210,450 |
| LR @ threshold 0.10 | — | — | — | -2,235,708 | 7,776,205 |

> The business cost ranking is completely different from the accuracy ranking. Random Forest scores 83% accuracy but is the second-worst model by dollar value, missing 62 of 71 real leavers.

### Optimal Threshold

Lowering the Logistic Regression threshold from 50% to **10%** reduces net business cost from 1,170,528 to -2,235,708 — a saving of 3,406,235 over the default. This is the recommended deployment configuration.

---

## Role-Specific Findings

Separate Random Forest models were trained for each of 4 role groups:

| Role Group | N | Attrition Rate | AUC |
|---|---|---|---|
| Admin/HR | 52 | 23.1% | 0.708 |
| Sales | 409 | 22.0% | 0.744 |
| Technical | 682 | 17.3% | 0.709 |
| Management | 327 | 5.2% | 0.463 |

### Top Drivers by Role Group

| Role Group | Driver 1 | Driver 2 | Driver 3 |
|---|---|---|---|
| Admin/HR | YearsAtCompany (0.110) | TotalWorkingYears (0.105) | MonthlyIncome (0.091) |
| Management | MonthlyIncome (0.115) | MonthlyRate (0.090) | HourlyRate (0.089) |
| Sales | DistanceFromHome (0.079) | MonthlyIncome (0.076) | DailyRate (0.072) |
| Technical | MonthlyIncome (0.098) | Age (0.076) | MonthlyRate (0.068) |

**MonthlyIncome is the top predictor in 3 of 4 role groups.** DistanceFromHome is uniquely critical for Sales — the only group where commute outranks pay.

---

## Action Plan

### Immediate Actions

1. **Eliminate or compensate overtime** — overtime triples attrition risk; focus on Sales Reps and Lab Technicians first
2. **Target the first 3 years of tenure** — 35% of under-1-year employees leave; implement structured onboarding with 6- and 12-month check-ins
3. **Raise bottom-quartile pay** — Q1 income group has 3x the attrition of Q4; targeted pay review for Sales and Technical roles
4. **Introduce hybrid flexibility for high-commute employees** — especially for Sales, where commute is the number 1 driver

### Medium-Term Actions

5. **Build a promotion pipeline** — YearsSinceLastPromotion is a significant risk factor; communicate clear advancement criteria
6. **Invest in Sales Representative retention** — 40% attrition rate and 0.388 average risk score; combine remote flexibility, pay floor increases, and territory redesign
7. **Monitor manager tenure** — YearsWithCurrManager ranks 9th in importance; track team-level attrition by manager

### Model Deployment

- **Deploy:** Logistic Regression at threshold 0.10 — best business value (saves 2.24M), fully interpretable
- **Secondary:** Naive Bayes for broad quarterly sweeps
- **Avoid:** KNN and default-threshold Random Forest — worst business outcomes despite reasonable accuracy

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook and paste each block in order
3. When Block 1 runs, it will prompt you to upload the CSV — upload `HR_Analytics_csv.csv`
4. Run blocks sequentially: Block 1 → 2 → 3 → 4 → 5

Each block depends on variables created by the previous block. Do not skip blocks or restart the runtime between them.

---

## Requirements

All dependencies are available by default in Google Colab. No installation needed.

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
google.colab
```

---

## Summary Stats

| Metric | Value |
|---|---|
| Total employees analyzed | 1,470 |
| Overall attrition rate | 16.1% |
| Models trained and compared | 6 |
| Critical-risk employees flagged | 171 |
| Estimated annual attrition cost | 5.54M |
| Savings from optimized LR model | 2.24M |
| Top attrition driver | MonthlyIncome |
| Highest-risk job role | Sales Representative (40%) |
| Overtime attrition multiplier | 3x |

---

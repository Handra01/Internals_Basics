# MLOps Lab CIE — TestGenAI Coverage Predictor

**Name:** Anushree Shetty
**USN:** 1BM23AI030

**Course:** MLOps (24AM6AEMLO)  
**College:** BMS College of Engineering  
**Semester:** VI — 2026 Even  

---

## Project Overview

This project predicts code coverage percentage for TestGenAI, a company that generates automated test suites using AI. The goal is to help the engineering team decide when human-written tests are still needed.

---

## Dataset

| Feature | Description |
|---|---|
| `codebase_size_kloc` | Size of codebase in KLOC (1–500) |
| `function_count` | Number of functions (10–1000) |
| `cyclomatic_complexity` | Code complexity score (1–20) |
| `is_legacy` | Whether the code is legacy (0 or 1) |
| `coverage_pct` | Target — code coverage percentage |

---

## Tasks

### Task 1 — Experiment Tracking & Model Comparison
- Trained LinearRegression and RandomForest
- Logged MAE, RMSE, R², MAPE to MLflow
- Best model selected by MAE → **LinearRegression**

### Task 2 — Hyperparameter Tuning
- Grid search over RandomForest params
- 3-fold cross-validation, 18 total trials
- Logged each trial as a nested MLflow run under `tuning-testgenai`

### Task 3 — Model Versioning
- Registered best model in MLflow Model Registry
- Model name: `testgenai-coverage-pct-predictor`

### Task 4 — Retraining Pipeline
- Combined original + new data
- Compared retrained vs champion on same test set
- Promoted if RMSE improvement ≥ 0.5

---

## Results

| Task | Output File |
|---|---|
| Task 1 | `results/step1_s1.json` |
| Task 2 | `results/step2_s2.json` |
| Task 3 | `results/step3_s6.json` |
| Task 4 | `results/step4_s8.json` |

---

## How to Run

```bash
cd MLOps_Lab_CIE
python src/train.py
python src/tune.py
python src/register_model.py
python src/retrain.py
```

---

## Tech Stack

- Python 3.x
- scikit-learn
- MLflow
- pandas, numpy

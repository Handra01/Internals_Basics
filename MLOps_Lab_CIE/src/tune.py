import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

DATA_PATH = "data/training_data.csv"


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("testgenai-coverage-pct")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    X = df[["codebase_size_kloc", "function_count", "cyclomatic_complexity", "is_legacy"]]
    y = df["coverage_pct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators":     [50, 100, 200],
        "max_depth":        [5, 10, None],
        "min_samples_split": [2, 5],
    }

    all_params = list(ParameterGrid(param_grid))
    total_trials = len(all_params)

    best_cv_mae   = float("inf")
    best_params   = None
    best_test_mae = None

    with mlflow.start_run(run_name="tuning-testgenai") as parent_run:
        mlflow.set_tag("project_phase", "hyperparameter_tuning")

        for params in all_params:
            model = RandomForestRegressor(random_state=42, **params)

            # 3-fold CV MAE on training set
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=3, scoring="neg_mean_absolute_error"
            )
            cv_mae = float(-cv_scores.mean())

            # Log each trial as a nested run
            with mlflow.start_run(nested=True, run_name=f"trial-n{params['n_estimators']}-d{params['max_depth']}-s{params['min_samples_split']}"):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mae", cv_mae)

            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_params = params

        # Train best model on full train set, evaluate on test set
        best_model = RandomForestRegressor(random_state=42, **best_params)
        best_model.fit(X_train, y_train)
        preds        = best_model.predict(X_test)
        best_test_mae = float(mean_absolute_error(y_test, preds))

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_test_mae", best_test_mae)
        mlflow.log_metric("best_cv_mae",   best_cv_mae)

        # Save tuned model so register_model.py can use it
        mlflow.sklearn.log_model(best_model, "tuned_model")

    output2 = {
        "search_type":      "grid",
        "n_folds":          3,
        "total_trials":     total_trials,
        "best_params":      {k: (v if v is not None else "None") for k, v in best_params.items()},
        "best_mae":         best_test_mae,
        "best_cv_mae":      best_cv_mae,
        "parent_run_name":  "tuning-testgenai",
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step2_s2.json", "w") as f:
        json.dump(output2, f, indent=4)

    print(f"Task 2 completed — {total_trials} trials, best cv_mae: {best_cv_mae:.4f}")


if __name__ == "__main__":
    main()
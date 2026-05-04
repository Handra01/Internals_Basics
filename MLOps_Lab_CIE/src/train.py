import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/training_data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def evaluate(model, name, X_train, X_test, y_train, y_test, run):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    # Log each model as a nested run under the parent
    with mlflow.start_run(run_id=run.info.run_id, nested=True):
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(model.get_params())   # all hyperparams
            mlflow.log_metric("mae",  mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2",   r2)
            mlflow.log_metric("mape", mape)
            mlflow.set_tag("project_phase", "model_selection")

    return mae, rmse, r2, mape


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("testgenai-coverage-pct")

    df = load_data()
    df.columns = df.columns.str.strip()

    X = df[["codebase_size_kloc", "function_count", "cyclomatic_complexity", "is_legacy"]]
    y = df["coverage_pct"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest",     RandomForestRegressor(random_state=42)),
    ]

    results = []

    with mlflow.start_run(run_name="model-selection") as run:
        mlflow.set_tag("project_phase", "model_selection")

        for name, model in models:
            mae, rmse, r2, mape = evaluate(
                model, name, X_train, X_test, y_train, y_test, run
            )
            results.append({
                "name": name,
                "mae":  float(mae),
                "rmse": float(rmse),
                "r2":   float(r2),
                "mape": float(mape),
            })

    best = min(results, key=lambda x: x["mae"])

    output1 = {
        "experiment_name":   "testgenai-coverage-pct",
        "models":            results,
        "best_model":        best["name"],
        "best_metric_name":  "mae",
        "best_metric_value": best["mae"],
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step1_s1.json", "w") as f:
        json.dump(output1, f, indent=4)

    print("Task 1 completed — best model:", best["name"])


if __name__ == "__main__":
    main()
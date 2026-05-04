import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH  = "data/training_data.csv"
NEW_PATH    = "data/new_data.csv"
RESULT_PATH = "results/step4_s8.json"


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("testgenai-coverage-pct")

    df_train = pd.read_csv(TRAIN_PATH)
    df_new   = pd.read_csv(NEW_PATH)

    df_train.columns = df_train.columns.str.strip()
    df_new.columns   = df_new.columns.str.strip()

    FEATURES = ["codebase_size_kloc", "function_count", "cyclomatic_complexity", "is_legacy"]

    # ── Fixed test set from original training data only ──────────────────────
    X_orig = df_train[FEATURES]
    y_orig = df_train["coverage_pct"]

    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X_orig, y_orig, test_size=0.2, random_state=42
    )

    # ── Champion: trained on original training split ──────────────────────────
    champion = RandomForestRegressor(random_state=42)
    champion.fit(X_train_orig, y_train_orig)
    champion_preds = champion.predict(X_test)
    champion_rmse  = float(np.sqrt(mean_squared_error(y_test, champion_preds)))

    # ── Retrained: trained on combined data, tested on same test set ──────────
    combined_df = pd.concat([df_train, df_new], ignore_index=True)
    X_combined  = combined_df[FEATURES]
    y_combined  = combined_df["coverage_pct"]

    # Use all of combined as training (test set is fixed from original split)
    retrained = RandomForestRegressor(random_state=42)
    retrained.fit(X_combined, y_combined)
    retrained_preds = retrained.predict(X_test)
    retrained_rmse  = float(np.sqrt(mean_squared_error(y_test, retrained_preds)))

    improvement = champion_rmse - retrained_rmse

    if improvement >= 0.5:
        action = "promoted"
    else:
        action = "kept_champion"

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="retraining-pipeline"):
        mlflow.log_metric("champion_rmse",  champion_rmse)
        mlflow.log_metric("retrained_rmse", retrained_rmse)
        mlflow.log_metric("improvement",    improvement)
        mlflow.log_param("action",          action)
        mlflow.set_tag("project_phase",     "retraining")

        if action == "promoted":
            mlflow.sklearn.log_model(retrained, "promoted_model")

    output = {
        "original_data_rows":      len(df_train),
        "new_data_rows":           len(df_new),
        "combined_data_rows":      len(combined_df),
        "champion_rmse":           champion_rmse,
        "retrained_rmse":          retrained_rmse,
        "improvement":             float(improvement),
        "min_improvement_threshold": 0.5,
        "action":                  action,
        "comparison_metric":       "rmse",
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Task 4 completed — action: {action} (improvement: {improvement:.4f})")


if __name__ == "__main__":
    main()
import os
import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

RESULT_PATH  = "results/step3_s6.json"
MODEL_NAME   = "testgenai-coverage-pct-predictor"   # exact name required by spec
EXPERIMENT   = "testgenai-coverage-pct"
PARENT_RUN   = "tuning-testgenai"                   # created by tune.py


def get_best_run():
    """Return the parent tuning run from tune.py so we register the tuned model."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT}' not found. Run tune.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{PARENT_RUN}'",
        order_by=["metrics.best_test_mae ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No run named '{PARENT_RUN}' found. Run tune.py first.")

    return runs[0]


def main():
    mlflow.set_tracking_uri("file:./mlruns")

    run = get_best_run()
    run_id     = run.info.run_id
    source_mae = run.data.metrics.get("best_test_mae", 0.0)

    model_uri = f"runs:/{run_id}/tuned_model"

    # Register in Model Registry
    registered = mlflow.register_model(model_uri, MODEL_NAME)

    output = {
        "registered_model_name": MODEL_NAME,
        "version":               int(registered.version),
        "run_id":                run_id,
        "source_metric":         "mae",
        "source_metric_value":   float(source_mae),
    }

    os.makedirs("results", exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Task 3 completed — registered {MODEL_NAME} v{registered.version}")


if __name__ == "__main__":
    main()
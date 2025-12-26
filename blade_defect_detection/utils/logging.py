"""MLflow logging utilities."""

from typing import Any, Dict

import mlflow


def log_metrics_to_mlflow(
    experiment: Any,
    run_id: str,
    metrics: Dict[str, Any],
) -> None:
    """Log metrics to MLflow.

    Args:
        experiment: MLflow experiment object
        run_id: MLflow run ID
        metrics: Dictionary of metrics to log
    """
    with mlflow.start_run(run_id=run_id):
        test_metrics_count = 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))
                if key.startswith("test_"):
                    test_metrics_count += 1

        if test_metrics_count > 0:
            print(f"Logged {test_metrics_count} test metric(s) to MLflow")


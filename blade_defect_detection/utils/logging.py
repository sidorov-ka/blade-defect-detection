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
        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, float(value))

        # Log hyperparameters (if available)
        # This will be done by Lightning's MLflowLogger automatically


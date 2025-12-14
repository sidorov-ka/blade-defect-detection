"""CLI commands for blade defect detection."""

from pathlib import Path

import fire
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from blade_defect_detection.training.train import train_model


def train(
    data_dir: str = None,
    config_path: str = "configs",
    config_name: str = "config",
):
    """Train the blade defect detection model.

    Args:
        data_dir: Path to data directory (overrides config)
        config_path: Path to configs directory
        config_name: Name of config file (without .yaml)
    """
    # Initialize Hydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)

    # Override data_dir if provided
    if data_dir:
        cfg.data.data_dir = data_dir

    # Convert to absolute path
    data_dir = Path(cfg.data.data_dir).resolve()

    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Train
    train_model(
        data_dir=data_dir,
        image_size=tuple(cfg.data.image_size),
        defect_classes=list(cfg.data.defect_classes),
        batch_size=cfg.dataloader.batch_size,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        num_workers=cfg.dataloader.num_workers,
        mlflow_tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=cfg.mlflow.run_name,
    )


if __name__ == "__main__":
    fire.Fire()


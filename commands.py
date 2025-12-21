"""CLI commands for blade defect detection."""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import fire
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from blade_defect_detection.inference.predict import predict_image
from blade_defect_detection.training.train import train_model


def _reset_hydra() -> None:
    """Utility to safely (re)initialize Hydra."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()


def _pull_dvc_data(data_path: Path) -> None:
    """Pull data from DVC if not available locally.

    Args:
        data_path: Path to data directory or file.
    """
    # If it's a directory – check that it contains at least one real image file.
    if data_path.exists() and data_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        has_images = any(
            p.is_file() and p.suffix.lower() in image_extensions
            for p in data_path.rglob("*")
        )
        if has_images:
            print(f"Data already exists at {data_path}, skipping DVC pull")
            return

    # If it's a regular file – consider it present and skip DVC pull.
    if data_path.exists() and data_path.is_file():
        print(f"Data file already exists at {data_path}, skipping DVC pull")
        return

    print(f"Pulling data from DVC to {data_path}...")
    try:
        # Load credentials from .env if available
        env = os.environ.copy()
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        if key.startswith("AWS_"):
                            env[key] = value
        
        # Pull specific data path from DVC
        result = subprocess.run(
            ["dvc", "pull", str(data_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        print("Data pulled successfully from DVC")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to pull data from DVC: {e.stderr}")
        print("Continuing with local data if available...")
    except FileNotFoundError:
        print("Warning: DVC not found. Make sure DVC is installed.")
        print("Continuing with local data if available...")


def train(
    data_dir: Optional[str] = None,
    config_path: str = "configs",
    config_name: str = "config",
) -> None:
    """Train the blade defect detection model.

    Args:
        data_dir: Path to data directory (overrides config)
        config_path: Path to configs directory
        config_name: Name of config file (without .yaml)
    """
    _reset_hydra()

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)

    # Override data_dir if provided
    if data_dir:
        cfg.data.data.data_dir = data_dir

    # Convert to absolute path
    data_dir_path = Path(cfg.data.data.data_dir).resolve()

    # Pull data from DVC if needed
    _pull_dvc_data(data_dir_path)

    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Generate run_name if not provided
    run_name = cfg.mlflow.mlflow.run_name
    if run_name is None:
        image_size = tuple(cfg.data.data.image_size)
        batch_size = cfg.data.dataloader.batch_size
        lr = cfg.training.training.learning_rate
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"img{image_size[0]}x{image_size[1]}_bs{batch_size}_lr{lr}_{timestamp}"
        print(f"Generated run_name: {run_name}")

    # Train
    train_model(
        data_dir=data_dir_path,
        image_size=tuple(cfg.data.data.image_size),
        defect_classes=list(cfg.data.data.defect_classes),
        batch_size=cfg.data.dataloader.batch_size,
        num_epochs=cfg.training.training.num_epochs,
        learning_rate=cfg.training.training.learning_rate,
        num_workers=cfg.data.dataloader.num_workers,
        mlflow_tracking_uri=cfg.mlflow.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.mlflow.experiment_name,
        run_name=run_name,
    )


def predict(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    output_path: Optional[str] = None,
    config_path: str = "configs",
    config_name: str = "config",
) -> str:
    """Run prediction on a single image and save visualization.

    Args:
        image_path: Path to input image
        checkpoint_path: Optional path to model checkpoint (.ckpt).
            If not provided, the latest checkpoint from models/ is used
        output_path: Optional path to save visualization.
            If not provided, saves to visualizations/pred_<stem>.png
        config_path: Hydra config path
        config_name: Hydra config name

    Returns:
        Path to saved visualization as string
    """
    _reset_hydra()

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)

    image_size = tuple(cfg.data.data.image_size)
    defect_classes = list(cfg.data.data.defect_classes)

    # Pull models from DVC if needed
    if checkpoint_path is None:
        models_dir = Path("models")
        if not models_dir.exists() or not list(models_dir.glob("*.ckpt")):
            _pull_dvc_data(models_dir)

    # Use predict_image from inference module
    result_path = predict_image(
        image_path=Path(image_path),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        image_size=image_size,
        defect_classes=defect_classes,
        output_path=Path(output_path) if output_path else None,
    )

    return str(result_path)


def main():
    """Main entry point."""
    fire.Fire()


if __name__ == "__main__":
    main()

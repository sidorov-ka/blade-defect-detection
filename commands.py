"""CLI commands for blade defect detection."""

import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen
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
    if data_path.exists() and data_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        has_images = any(
            p.is_file() and p.suffix.lower() in image_extensions
            for p in data_path.rglob("*")
        )
        if has_images:
            print(f"Data already exists at {data_path}, skipping DVC pull")
            return

    if data_path.exists() and data_path.is_file():
        print(f"Data file already exists at {data_path}, skipping DVC pull")
        return

    print(f"Pulling data from DVC to {data_path}...")
    max_retries = 3
    try:
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    print(f"\nRetry attempt {attempt}/{max_retries}...")
                
                process = Popen(
                    ["uv", "run", "dvc", "pull", str(data_path)],
                    stdout=PIPE,
                    stderr=PIPE,
                    text=True,
                    bufsize=1,
                )

                if attempt == 1:
                    print("Downloading data (this may take several minutes for large datasets)...")
                print("Progress: ", end="", flush=True)

                def show_progress():
                    """Show animated progress dots."""
                    dots = ["", ".", "..", "..."]
                    i = 0
                    while process.poll() is None:
                        print(f"\rProgress: {dots[i % len(dots)]}", end="", flush=True)
                        i += 1
                        time.sleep(0.5)

                progress_thread = threading.Thread(target=show_progress, daemon=True)
                progress_thread.start()

                stdout, stderr = process.communicate()

                print("\r" + " " * 20 + "\r", end="")

                if process.returncode == 0:
                    print("✓ Data pulled successfully from DVC")
                    return
                else:
                    error_msg = stderr if stderr else stdout
                    if attempt < max_retries:
                        print(f"\n⚠ Attempt {attempt} failed, retrying...")
                        time.sleep(2)
                        continue
                    raise subprocess.CalledProcessError(
                        process.returncode, "dvc pull", stderr=error_msg
                    )
            except FileNotFoundError:
                # If uv is not found, no point in retrying
                raise
            except subprocess.CalledProcessError as e:
                if attempt < max_retries:
                    continue
                raise

    except subprocess.CalledProcessError as e:
        print(f"\n⚠ Warning: Failed to pull data from DVC: {e.stderr if hasattr(e, 'stderr') else str(e)}")
        print("You can try running manually: uv run dvc pull")
        print("Continuing with local data if available...")
    except FileNotFoundError:
        print("\n⚠ Warning: uv or DVC not found. Make sure uv is installed and DVC is in dependencies.")
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

    if data_dir:
        cfg.data.data.data_dir = data_dir

    data_dir_path = Path(cfg.data.data.data_dir).resolve()
    _pull_dvc_data(data_dir_path)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    run_name = cfg.mlflow.mlflow.run_name
    if run_name is None:
        image_size = tuple(cfg.data.data.image_size)
        batch_size = cfg.data.dataloader.batch_size
        lr = cfg.training.training.learning_rate
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"img{image_size[0]}x{image_size[1]}_bs{batch_size}_lr{lr}_{timestamp}"
        print(f"Generated run_name: {run_name}")

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

    if checkpoint_path is None:
        models_dir = Path("models")
        if not models_dir.exists() or not list(models_dir.glob("*.ckpt")):
            _pull_dvc_data(models_dir)

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

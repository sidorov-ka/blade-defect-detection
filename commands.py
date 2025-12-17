"""CLI commands for blade defect detection."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fire
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from blade_defect_detection.training.train import (
    BladeDefectLightningModule,
    _mask_to_color,
    train_model,
)


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
        run_name=cfg.mlflow.mlflow.run_name,
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
        image_path: Path to input image.
        checkpoint_path: Optional path to model checkpoint (.ckpt). If not provided,
            the latest checkpoint from models/ is used.
        output_path: Optional path to save visualization. If not provided,
            saves to models/pred_<stem>.png.
        config_path: Hydra config path.
        config_name: Hydra config name.

    Returns:
        Path to saved visualization as string.
    """
    _reset_hydra()

    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)

    image_size = tuple(cfg.data.data.image_size)
    defect_classes = list(cfg.data.data.defect_classes)
    class_names = ["background"] + defect_classes

    # Resolve checkpoint
    if checkpoint_path is None:
        models_dir = Path("models")
        # Pull models from DVC if needed
        if not models_dir.exists() or not list(models_dir.glob("*.ckpt")):
            _pull_dvc_data(models_dir)
        ckpts = sorted(models_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found in models/ directory")
        checkpoint = ckpts[-1]
    else:
        checkpoint = Path(checkpoint_path)

    print(f"Using checkpoint: {checkpoint}")

    # Load model from checkpoint
    model = BladeDefectLightningModule.load_from_checkpoint(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Load and preprocess image (same basic logic as dataset)
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.BILINEAR)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu()

    # Build simple palette (same as training visualization)
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    pred_color = _mask_to_color(pred_mask, palette)
    pred_overlay = Image.blend(img.convert("RGB"), pred_color, alpha=0.4)

    # Determine main predicted class (excluding background)
    if pred_mask.max() > 0:
        pred_major = int(pred_mask[pred_mask > 0].mode()[0])
    else:
        pred_major = 0
    pred_label = (
        class_names[pred_major] if pred_major < len(class_names) else str(pred_major)
    )

    # Add caption
    canvas = Image.new(
        "RGB", (pred_overlay.width, pred_overlay.height + 30), (255, 255, 255)
    )
    canvas.paste(pred_overlay, (0, 0))

    caption = f"Predicted: {pred_label}"
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((canvas.width - text_w) // 2, canvas.height - text_h - 5),
        caption,
        fill=(0, 0, 0),
        font=font,
    )

    # Save result
    if output_path is None:
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pred_{Path(image_path).stem}.png"
    else:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas.save(out_path)
    print(f"Saved prediction visualization to: {out_path}")
    return str(out_path)


def main():
    """Main entry point."""
    fire.Fire()


if __name__ == "__main__":
    main()

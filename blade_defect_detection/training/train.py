"""PyTorch Lightning module for training and inference."""

from pathlib import Path
from typing import Optional, Sequence

import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics import JaccardIndex, MetricCollection
from torchvision import transforms

from blade_defect_detection.data.dataset import BladeDefectDataset
from blade_defect_detection.models.model import BladeDefectModel
from blade_defect_detection.utils.logging import log_metrics_to_mlflow


class BladeDefectLightningModule(LightningModule):
    """Lightning module for blade defect segmentation."""

    def __init__(
        self,
        num_classes: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize Lightning module.

        Args:
            num_classes: Number of classes (background + defects)
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Model
        self.model = BladeDefectModel(num_classes=num_classes)

        # Loss function (CrossEntropy for multi-class segmentation)
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_metrics = MetricCollection(
            {
                "train_iou": JaccardIndex(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_iou": JaccardIndex(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )
        self.test_metrics = MetricCollection(
            {
                "test_iou": JaccardIndex(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, masks = batch

        # Validate masks
        mask_min, mask_max = masks.min().item(), masks.max().item()
        if mask_min < 0 or mask_max >= self.num_classes:
            raise ValueError(
                f"Invalid mask values: min={mask_min}, max={mask_max}, "
                f"expected range [0, {self.num_classes-1}]"
            )

        logits = self(images)
        loss = self.criterion(logits, masks)

        # Check for NaN or invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at step {batch_idx}: {loss.item()}")
            print(
                f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                f"mean={logits.mean().item():.4f}"
            )
            print(
                f"  Masks: min={mask_min}, max={mask_max}, unique={torch.unique(masks).tolist()}"
            )
            # Use a small positive value instead of NaN
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)

        # Compute predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.train_metrics(preds, masks)

        # Log
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        images, masks = batch

        # Validate masks
        mask_min, mask_max = masks.min().item(), masks.max().item()
        if mask_min < 0 or mask_max >= self.num_classes:
            raise ValueError(
                f"Invalid mask values: min={mask_min}, max={mask_max}, "
                f"expected range [0, {self.num_classes-1}]"
            )

        logits = self(images)
        loss = self.criterion(logits, masks)

        # Check for NaN or invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at val step {batch_idx}: {loss.item()}")
            loss = torch.tensor(1.0, device=loss.device, requires_grad=True)

        # Compute predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_metrics(preds, masks)

        # Log
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step mirrors validation for reporting."""
        images, masks = batch
        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)
        self.test_metrics(preds, masks)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


def train_model(
    data_dir: Path,
    image_size: tuple = (256, 256),
    defect_classes: Optional[list] = None,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    mlflow_tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "blade_defect_detection",
    run_name: Optional[str] = None,
):
    """Train the model.

    Args:
        data_dir: Root directory containing images/ and masks/
        image_size: Target image size (height, width)
        defect_classes: List of defect class names (e.g., ['dent', 'nick', 'scratch', 'corrosion'])
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        num_workers: Number of data loading workers
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: MLflow experiment name
        run_name: MLflow run name (optional)
    """
    data_dir = Path(data_dir)

    # Check if data is split into train/val/test or needs automatic splitting
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        # Use separate train/val/test directories
        train_dataset = BladeDefectDataset(
            train_dir, image_size=image_size, defect_classes=defect_classes
        )
        val_dataset = BladeDefectDataset(
            val_dir, image_size=image_size, defect_classes=defect_classes
        )
        test_dataset = BladeDefectDataset(
            test_dir, image_size=image_size, defect_classes=defect_classes
        )
    else:
        # Create single dataset and split automatically (70/15/15)
        full_dataset = BladeDefectDataset(
            data_dir, image_size=image_size, defect_classes=defect_classes
        )
        total_len = len(full_dataset)
        train_size = max(1, int(0.7 * total_len))
        val_size = max(1, int(0.15 * total_len))
        test_size = total_len - train_size - val_size
        if test_size <= 0:
            test_size = 1
            train_size = max(1, train_size - 1)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Get number of classes from dataset
    # If using random_split, access the underlying dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        num_classes = train_dataset.dataset.num_classes
    else:
        num_classes = train_dataset.num_classes

    # Create model
    model = BladeDefectLightningModule(
        num_classes=num_classes,
        learning_rate=learning_rate,
    )

    # Setup loggers
    loggers = []

    # TensorBoard logger for easy visualization (always enabled)
    # Убираем name, чтобы версии инкрементировались правильно в lightning_logs/version_X/
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs",
    )
    loggers.append(tensorboard_logger)

    # MLflow logger (optional - only if server is available)
    mlflow_logger = None
    try:
        # Check if MLflow server is reachable before creating logger
        if mlflow_tracking_uri.startswith("http"):
            try:
                # Try to connect to MLflow server with timeout
                health_url = mlflow_tracking_uri.rstrip("/") + "/health"
                response = requests.get(health_url, timeout=3)
                if response.status_code != 200:
                    raise ConnectionError(f"MLflow server returned status {response.status_code}")
            except (requests.exceptions.RequestException, ConnectionError) as e:
                print(f"MLflow server not reachable ({e}), using TensorBoard only")
                mlflow_logger = None
            else:
                # Server is reachable, create logger
                mlflow_logger = MLFlowLogger(
                    experiment_name=experiment_name,
                    tracking_uri=mlflow_tracking_uri,
                    run_name=run_name,
                )
                loggers.append(mlflow_logger)
                print("MLflow logging enabled")
        else:
            # File-based tracking, try to create logger
            mlflow_logger = MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow_tracking_uri,
                run_name=run_name,
            )
            loggers.append(mlflow_logger)
            print("MLflow logging enabled")
    except Exception as e:
        print(f"MLflow logger initialization failed ({e}), using TensorBoard only")
        mlflow_logger = None

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="models",
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )

    # Trainer with GPU support
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=loggers,
        callbacks=[checkpoint_callback],
        accelerator="gpu",  # Explicitly use GPU
        devices=1,  # Use single GPU
        precision="32",  # Full precision (16-mixed has issues with some optimizers)
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train + test
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Log additional metrics to MLflow (if available)
    if mlflow_logger is not None:
        try:
            log_metrics_to_mlflow(
                mlflow_logger.experiment,
                mlflow_logger.run_id,
                trainer.callback_metrics,
            )
        except Exception as e:
            print(f"Failed to log metrics to MLflow: {e}")

    return trainer, model


def _mask_to_color(mask: torch.Tensor, palette: Sequence[tuple[int, int, int]]) -> Image.Image:
    """Convert mask tensor [H, W] to a color PIL image using palette."""
    mask_np = mask.cpu().numpy()
    h, w = mask_np.shape
    color_img = Image.new("RGB", (w, h))
    pixels = color_img.load()
    for i in range(h):
        for j in range(w):
            cls = int(mask_np[i, j])
            color = palette[cls] if cls < len(palette) else (255, 255, 255)
            pixels[j, i] = color
    return color_img


def predict_image(
    image_path: Path,
    checkpoint_path: Optional[Path] = None,
    image_size: tuple = (256, 256),
    defect_classes: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Run prediction for a single image and save visualization.

    Args:
        image_path: Path to input image.
        checkpoint_path: Path to .ckpt file. If None, latest from models/ is used.
        image_size: Target image size (height, width).
        defect_classes: List of defect class names.
        output_path: Where to save visualization. Defaults to `visualizations/<stem>_pred.png`.
    """
    image_path = Path(image_path)

    # Resolve checkpoint
    if checkpoint_path is None:
        models_dir = Path("models")
        ckpts = sorted(models_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            raise FileNotFoundError("No checkpoints found in models/ directory")
        checkpoint_path = ckpts[0]
    else:
        checkpoint_path = Path(checkpoint_path)

    # Load model from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BladeDefectLightningModule.load_from_checkpoint(str(checkpoint_path))
    model.to(device)
    model.eval()

    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size, Image.BILINEAR)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu()

    # Prepare visualization
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    class_names = ["background"] + list(
        defect_classes or ["dent", "nick", "scratch", "corrosion"]
    )

    base_image = img
    pred_color = _mask_to_color(pred_mask, palette)
    pred_overlay = Image.blend(base_image.convert("RGB"), pred_color, alpha=0.4)

    # Major predicted class (excluding background)
    if pred_mask.max() > 0:
        pred_major = int(pred_mask[pred_mask > 0].mode()[0])
    else:
        pred_major = 0
    pred_label = (
        class_names[pred_major] if pred_major < len(class_names) else str(pred_major)
    )

    # Create canvas with caption
    combined_height = pred_overlay.height + 30
    canvas = Image.new("RGB", (pred_overlay.width, combined_height), (255, 255, 255))
    canvas.paste(pred_overlay, (0, 0))

    caption = f"Predicted: {pred_label}"
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((pred_overlay.width - text_w) // 2, combined_height - text_h - 5),
        caption,
        fill=(0, 0, 0),
        font=font,
    )

    if output_path is None:
        # Сохраняем в visualizations/ вместо рядом с исходным изображением
        vis_dir = Path("visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_path = vis_dir / f"{image_path.stem}_pred.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return output_path

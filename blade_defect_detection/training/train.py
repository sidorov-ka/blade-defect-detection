"""PyTorch Lightning module for training."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchmetrics import JaccardIndex, MetricCollection

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
                "train_iou": JaccardIndex(task="multiclass", num_classes=num_classes, average="macro"),
            }
        )
        self.val_metrics = MetricCollection(
            {
                "val_iou": JaccardIndex(task="multiclass", num_classes=num_classes, average="macro"),
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
            print(f"  Logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
            print(f"  Masks: min={mask_min}, max={mask_max}, unique={torch.unique(masks).tolist()}")
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

    # Check if data is split into train/val or needs automatic splitting
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists() and val_dir.exists():
        # Use separate train/val directories
        train_dataset = BladeDefectDataset(
            train_dir, image_size=image_size, defect_classes=defect_classes
        )
        val_dataset = BladeDefectDataset(
            val_dir, image_size=image_size, defect_classes=defect_classes
        )
    else:
        # Create single dataset and split automatically (80/20)
        full_dataset = BladeDefectDataset(
            data_dir, image_size=image_size, defect_classes=defect_classes
        )
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
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
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="blade_defect_detection",
    )
    loggers.append(tensorboard_logger)
    
    # MLflow logger (optional - only if server is available)
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_name=run_name,
        )
        # Test connection
        _ = mlflow_logger.experiment
        loggers.append(mlflow_logger)
        print("MLflow logging enabled")
    except Exception as e:
        print(f"MLflow server not available ({e}), using TensorBoard only")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.2f}",
    )

    # Trainer with GPU support
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=loggers,
        callbacks=[checkpoint_callback],
        accelerator="gpu",  # Explicitly use GPU
        devices=1,  # Use single GPU
        precision="16-mixed",  # Mixed precision for faster training and less memory
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Log additional metrics to MLflow
    log_metrics_to_mlflow(
        mlflow_logger.experiment,
        mlflow_logger.run_id,
        trainer.callback_metrics,
    )

    return trainer, model


"""PyTorch Lightning module for training and inference."""

import subprocess
from pathlib import Path
from typing import Optional, Sequence

import mlflow
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

    def dice_loss(
        self, pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6
    ) -> torch.Tensor:
        """Compute Dice Loss for multi-class segmentation.

        Args:
            pred: Logits tensor [B, num_classes, H, W]
            target: Target masks [B, H, W] with class indices
            num_classes: Number of classes
            smooth: Smoothing factor to avoid division by zero

        Returns:
            Dice loss value (scalar tensor)
        """
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        dice_scores = []
        for i in range(num_classes):
            pred_i = pred_softmax[:, i]
            target_i = target_one_hot[:, i]
            intersection = (pred_i * target_i).sum(dim=(1, 2))  # Sum over H, W
            union = pred_i.sum(dim=(1, 2)) + target_i.sum(dim=(1, 2))
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)

        # Average over classes and batches
        dice_loss = 1 - torch.stack(dice_scores).mean()
        return dice_loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, masks = batch

        # Validate masks (with warning instead of error to not stop training)
        mask_min, mask_max = masks.min().item(), masks.max().item()
        if mask_min < 0 or mask_max >= self.num_classes:
            print(
                f"WARNING: Invalid mask values at train step {batch_idx}: "
                f"min={mask_min}, max={mask_max}, "
                f"expected range [0, {self.num_classes-1}]. "
                f"Clamping to valid range."
            )
            # Clamp masks to valid range instead of raising error
            masks = torch.clamp(masks, 0, self.num_classes - 1)

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

        # Validate masks (with warning instead of error to not stop training)
        mask_min, mask_max = masks.min().item(), masks.max().item()
        if mask_min < 0 or mask_max >= self.num_classes:
            print(
                f"WARNING: Invalid mask values at val step {batch_idx}: "
                f"min={mask_min}, max={mask_max}, "
                f"expected range [0, {self.num_classes-1}]. "
                f"Clamping to valid range."
            )
            # Clamp masks to valid range instead of raising error
            masks = torch.clamp(masks, 0, self.num_classes - 1)

        logits = self(images)
        
        # Combined loss: CrossEntropy + Dice Loss
        loss_ce = self.criterion_ce(logits, masks)
        loss_dice = self.dice_loss(logits, masks, self.num_classes)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        
        # Log individual losses for monitoring
        self.log("val_loss_ce", loss_ce, on_step=False, on_epoch=True)
        self.log("val_loss_dice", loss_dice, on_step=False, on_epoch=True)

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
        loss_ce = self.criterion_ce(logits, masks)
        loss_dice = self.dice_loss(logits, masks, self.num_classes)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        preds = torch.argmax(logits, dim=1)
        self.test_metrics(preds, masks)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler: reduce LR when validation loss plateaus
        # This helps fine-tune the model and improve convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Minimize validation loss
            factor=0.5,  # Reduce LR by half
            patience=3,  # Wait 3 epochs without improvement
            verbose=True,  # Print when LR is reduced
            min_lr=1e-6,  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Monitor validation loss
                "interval": "epoch",  # Update every epoch
                "frequency": 1,
            },
        }


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
        if total_len == 0:
            raise ValueError(
                f"No samples found in data_dir={data_dir}. "
                "Make sure data is downloaded via DVC "
                "(e.g., `uv run dvc pull` or `uv run dvc checkout data.dvc`) "
                "and the directory structure matches the README."
            )
        train_size = max(1, int(0.7 * total_len))
        val_size = max(1, int(0.15 * total_len))
        test_size = total_len - train_size - val_size
        if test_size <= 0:
            test_size = 1
            train_size = max(1, train_size - 1)
        train_subset, val_subset, test_subset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        # Create train dataset with augmentation using train subset indices
        # Create a new dataset instance with augmentation and filter samples
        train_dataset = BladeDefectDataset(
            data_dir,
            image_size=image_size,
            defect_classes=defect_classes,
            transform=train_transform,
        )
        # Filter samples to match train subset
        train_indices = set(train_subset.indices)
        train_dataset.samples = [
            sample for i, sample in enumerate(full_dataset.samples) if i in train_indices
        ]
        
        # Create val and test datasets without augmentation
        val_dataset = BladeDefectDataset(
            data_dir, image_size=image_size, defect_classes=defect_classes
        )
        val_indices = set(val_subset.indices)
        val_dataset.samples = [
            sample for i, sample in enumerate(full_dataset.samples) if i in val_indices
        ]
        
        test_dataset = BladeDefectDataset(
            data_dir, image_size=image_size, defect_classes=defect_classes
        )
        test_indices = set(test_subset.indices)
        test_dataset.samples = [
            sample for i, sample in enumerate(full_dataset.samples) if i in test_indices
        ]

    # Check dataset sizes before creating dataloaders
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty! Cannot train without validation data. "
            "Check your data directory structure."
        )
    
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty! Cannot train without training data. "
            "Check your data directory structure."
        )

    # Create dataloaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # Reuse workers
        prefetch_factor=2,  # Reduce prefetching to save memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,
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
    # Logs will be in lightning_logs/version_X/
    tensorboard_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=None,  # No additional lightning_logs/ subfolder
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
                # Server is reachable, ensure experiment exists before creating logger
                try:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                    # Get or create experiment to avoid race condition
                    try:
                        experiment = mlflow.get_experiment_by_name(experiment_name)
                        if experiment is None:
                            # Experiment doesn't exist, create it
                            experiment_id = mlflow.create_experiment(experiment_name)
                            print(
                                f"Created MLflow experiment: {experiment_name} "
                                f"(ID: {experiment_id})"
                            )
                        elif experiment.lifecycle_stage == "deleted":
                            # Experiment exists but is deleted
                            # Try to restore it, if that fails, delete permanently and create new
                            from mlflow.tracking import MlflowClient
                            client = MlflowClient(mlflow_tracking_uri)
                            try:
                                client.restore_experiment(experiment.experiment_id)
                                experiment_id = experiment.experiment_id
                                print(
                                    f"Restored deleted MLflow experiment: "
                                    f"{experiment_name} (ID: {experiment_id})"
                                )
                            except Exception:
                                # If restore fails, delete permanently and create new
                                try:
                                    client.delete_experiment(experiment.experiment_id)
                                except Exception:
                                    pass  # Ignore if already deleted
                                experiment_id = mlflow.create_experiment(experiment_name)
                                print(
                                    f"Created new MLflow experiment (old was deleted): "
                                    f"{experiment_name} (ID: {experiment_id})"
                                )
                        else:
                            # Experiment exists and is active
                            experiment_id = experiment.experiment_id
                            print(
                                f"Using existing MLflow experiment: "
                                f"{experiment_name} (ID: {experiment_id})"
                            )
                    except Exception as exp_err:
                        # If get_experiment_by_name fails, try to create
                        try:
                            experiment_id = mlflow.create_experiment(experiment_name)
                            print(
                                f"Created MLflow experiment: {experiment_name} "
                                f"(ID: {experiment_id})"
                            )
                        except Exception as create_err:
                            raise Exception(
                                f"Failed to get/create experiment: "
                                f"get={exp_err}, create={create_err}"
                            )
                    
                    # Set active experiment to prevent MLFlowLogger from trying to create it
                    mlflow.set_experiment(experiment_name)
                    print(f"Set active experiment: {experiment_name}")
                    
                except Exception as e:
                    print(f"Failed to setup MLflow experiment ({e}), using TensorBoard only")
                    import traceback
                    traceback.print_exc()
                    mlflow_logger = None
                else:
                    # Create logger - experiment is already set as active
                    try:
                        mlflow_logger = MLFlowLogger(
                            experiment_name=experiment_name,
                            tracking_uri=mlflow_tracking_uri,
                            run_name=run_name,
                        )
                        loggers.append(mlflow_logger)
                        print(f"MLflow logging enabled for run: {run_name}")
                    except Exception as e:
                        print(f"MLflow logger creation failed ({e}), using TensorBoard only")
                        import traceback
                        traceback.print_exc()
                        mlflow_logger = None
        else:
            # File-based tracking, ensure experiment exists
            try:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
            except Exception:
                pass  # Ignore errors for file-based tracking
            
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

    # Log git commit id to MLflow (if available)
    # Note: This is done after logger is created, so run_id should be available
    if mlflow_logger is not None and hasattr(mlflow_logger, 'run_id') and mlflow_logger.run_id:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                cwd=Path.cwd(),
            )
            git_commit_id = result.stdout.strip() if result.returncode == 0 else "unknown"
            
            # Set tracking URI before logging
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            with mlflow.start_run(run_id=mlflow_logger.run_id):
                mlflow.set_tag("git_commit_id", git_commit_id)
            print(f"Logged git commit id to MLflow: {git_commit_id}")
        except Exception as e:
            # Silently ignore errors - this is not critical
            pass

    # Callbacks
    # ModelCheckpoint monitors val_loss - will save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="models",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_last=True,  # Also save last checkpoint
        verbose=True,  # Print checkpoint info
    )
    
    # Early stopping: stop training if validation loss doesn't improve
    # This prevents overfitting and saves training time
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,  # Minimum change to qualify as improvement
        patience=5,  # Number of epochs to wait before stopping
        mode="min",  # Minimize validation loss
        verbose=True,  # Print when early stopping is triggered
        check_finite=True,  # Stop if loss becomes NaN or Inf
    )

    # Trainer with GPU support and memory optimizations
    # Use gradient accumulation to emulate larger batch size
    # With batch_size=6 and accumulate_grad_batches=6, effective batch size = 36
    # Note: Using full precision (32-bit) due to NaN issues with mixed precision
    # and gradient accumulation. Memory is saved through reduced model size.
    accumulate_grad_batches = 6
    trainer = Trainer(
        max_epochs=num_epochs,
        min_epochs=1,  # Minimum epochs to train
        logger=loggers,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="gpu",  # Explicitly use GPU
        devices=1,  # Use single GPU
        precision="32",  # Full precision (16-mixed causes NaN with gradient accumulation)
        accumulate_grad_batches=accumulate_grad_batches,  # Accumulate gradients
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,  # Show model summary
        check_val_every_n_epoch=1,  # Validate every epoch
        val_check_interval=1.0,  # Validate after each training epoch
        gradient_clip_val=1.0,  # Gradient clipping to prevent NaN
    )

    # Print dataloader info
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {accumulate_grad_batches}")
    print(f"  Effective batch size (with accumulation): {batch_size * accumulate_grad_batches}")
    print(f"  Precision: 32-bit (full precision for stability)")
    print(f"  Model: Reduced channels (48->96->192->384->768)")
    print(f"  Loss: Combined CrossEntropy + Dice Loss (optimizes IoU)")
    print(f"  Augmentation: Color jitter (brightness, contrast, saturation, hue)")
    print(f"  LR Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    print(f"  Early Stopping: Enabled (patience=5)")
    
    # Warn if dataset is very small
    if len(train_loader) < 5:
        print(f"\nWARNING: Training dataset is very small ({len(train_loader)} batches).")
        print("This may cause training to complete very quickly.")
    if len(val_loader) == 0:
        raise ValueError("Validation dataloader is empty! Cannot train without validation data.")
    
    # Train + validation
    print(f"\nStarting training for {num_epochs} epochs...")
    print(
        f"Expected training time: "
        f"~{len(train_loader) * num_epochs / 60:.1f} minutes (rough estimate)"
    )
    try:
        print(f"Calling trainer.fit() for {num_epochs} epochs...")
        trainer.fit(model, train_loader, val_loader)
        actual_epochs = trainer.current_epoch + 1
        print(f"\n✓ Training completed successfully after {actual_epochs} epochs!")
        print(
            f"Trainer state: current_epoch={trainer.current_epoch}, "
            f"max_epochs={trainer.max_epochs}"
        )
        if actual_epochs < num_epochs:
            print(
                f"⚠ WARNING: Training stopped after {actual_epochs} epochs "
                f"instead of {num_epochs}!"
            )
            print("This may indicate an issue. Check logs above for errors.")
    except KeyboardInterrupt:
        print(f"\n⚠ Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: Training failed with exception: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - continue to test if possible
        print("Continuing to test evaluation despite training error...")
    finally:
        print(
            f"After trainer.fit(): current_epoch={trainer.current_epoch}, "
            f"max_epochs={trainer.max_epochs}"
        )

    # Test on test set (always run test, even if training stopped early)
    # This helps with debugging and we still get test metrics
    print(f"\nRunning test evaluation...")
    print(f"Training completed {trainer.current_epoch + 1} out of {num_epochs} epochs")
    print("\n" + "=" * 50)
    print("Running evaluation on test set...")
    print("=" * 50)
    test_results = None
    try:
        test_results = trainer.test(model, test_loader)
        print("Test evaluation completed successfully")
    except Exception as e:
        print(f"WARNING: Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        test_results = None
    
    # Print test results
    if test_results:
        print("\nTest Results:")
        for key, value in test_results[0].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

    # Log test metrics explicitly to MLflow (if available)
    # Wrap in try-except to prevent errors from stopping the function
    print(f"\nLogging metrics to MLflow...")
    if mlflow_logger is not None:
        print(f"MLflow logger available, run_id: {mlflow_logger.run_id}")
        try:
            # Log all callback metrics (includes test metrics after test())
            log_metrics_to_mlflow(
                mlflow_logger.experiment,
                mlflow_logger.run_id,
                trainer.callback_metrics,
            )
            print("✓ Callback metrics logged to MLflow")
        except Exception as e:
            print(f"⚠ WARNING: Failed to log callback metrics to MLflow: {e}")
            import traceback
            traceback.print_exc()
            # Continue - don't let MLflow errors stop the process
        
        try:
            # Also explicitly log test results if available
            if test_results:
                print("Logging test results to MLflow...")
                with mlflow.start_run(run_id=mlflow_logger.run_id):
                    for key, value in test_results[0].items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, float(value))
                print("✓ Test metrics logged to MLflow")
        except Exception as e:
            print(f"⚠ WARNING: Failed to log test metrics to MLflow: {e}")
            import traceback
            traceback.print_exc()
            # Continue - don't let MLflow errors stop the process
    else:
        print("MLflow logger not available, skipping MLflow logging")

    print(f"\n✓ Training function completed successfully.")
    print(f"  - Training epochs: {trainer.current_epoch + 1}/{num_epochs}")
    print(f"  - Test evaluation: {'Completed' if test_results else 'Skipped/Failed'}")
    print(f"  - MLflow logging: {'Completed' if mlflow_logger else 'Not available'}")
    return trainer, model



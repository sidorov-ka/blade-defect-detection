"""Dataset class for blade defect detection."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BladeDefectDataset(Dataset):
    """Dataset for blade defect segmentation.

    Loads images and masks from organized directories:
    - data/images/{class}/ - images
    - data/masks/{class}/ - masks (only for defect classes)
    """

    def __init__(
        self,
        data_dir: Path,
        image_size: Tuple[int, int] = (256, 256),
        defect_classes: Optional[list] = None,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing images/ and masks/ folders
            image_size: Target size for resizing (height, width)
            defect_classes: List of defect class names
                (e.g., ['dent', 'nick', 'scratch', 'corrosion'])
            transform: Optional torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.defect_classes = defect_classes or ["dent", "nick", "scratch", "corrosion"]
        self.transform = transform

        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.defect_classes)}
        self.num_classes = len(self.defect_classes) + 1
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list:
        """Collect all image-mask pairs from data directory."""
        samples = []

        images_dir = self.data_dir / "images"
        masks_dir = self.data_dir / "masks"

        # Collect defect samples with masks
        for defect_class in self.defect_classes:
            class_images_dir = images_dir / defect_class
            class_masks_dir = masks_dir / defect_class

            if not class_images_dir.exists():
                continue

            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            for image_path in class_images_dir.iterdir():
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in image_extensions:
                    continue

                mask_path = class_masks_dir / image_path.name
                if mask_path.exists() and mask_path.is_file():
                    samples.append(
                        {
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "class_idx": self.class_to_idx[defect_class],
                        }
                    )

        # Collect normal samples (no masks, all background)
        normal_dir = images_dir / "normal"
        if normal_dir.exists():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            for image_path in normal_dir.iterdir():
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in image_extensions:
                    continue
                samples.append(
                    {
                        "image_path": image_path,
                        "mask_path": None,
                        "class_idx": 0,
                    }
                )

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get image and mask pair.

        Returns:
            Tuple of (image_tensor, mask_tensor)
            - image_tensor: [3, H, W] float32 in [0, 1]
            - mask_tensor: [H, W] int64 with class indices [0, 1, 2, 3, 4]
        """
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize(self.image_size, Image.BILINEAR)

        # Load or create mask
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = mask.resize(self.image_size, Image.NEAREST)
            mask_array = np.array(mask, dtype=np.uint8)

            if mask_array.max() <= 1:
                mask_array = np.where(mask_array > 0.5, sample["class_idx"], 0)
            else:
                mask_array = np.where(mask_array > 127, sample["class_idx"], 0)

            mask_array = mask_array.astype(np.int64)

            if mask_array.min() < 0 or mask_array.max() >= self.num_classes:
                print(
                    f"WARNING: Invalid mask values in {sample['mask_path']}: "
                    f"min={mask_array.min()}, max={mask_array.max()}, "
                    f"expected range [0, {self.num_classes-1}]. Clamping to valid range."
                )
                mask_array = np.clip(mask_array, 0, self.num_classes - 1)
        else:
            mask_array = np.zeros(self.image_size, dtype=np.int64)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = torch.from_numpy(mask_array).long()

        if mask_tensor.min() < 0 or mask_tensor.max() >= self.num_classes:
            print(
                f"WARNING: Mask validation failed: min={mask_tensor.min()}, "
                f"max={mask_tensor.max()}, num_classes={self.num_classes}. "
                f"Clamping to valid range."
            )
            mask_tensor = torch.clamp(mask_tensor, 0, self.num_classes - 1)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, mask_tensor


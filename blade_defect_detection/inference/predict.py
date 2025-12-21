"""Prediction and inference utilities for blade defect detection."""

from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from blade_defect_detection.training.train import BladeDefectLightningModule


def mask_to_color(
    mask: torch.Tensor, palette: Sequence[tuple[int, int, int]]
) -> Image.Image:
    """Convert mask tensor [H, W] to a color PIL image using palette.

    Args:
        mask: Mask tensor with class indices [H, W]
        palette: List of RGB tuples for each class

    Returns:
        Color PIL image
    """
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
    image_size: tuple = (192, 192),
    defect_classes: Optional[Sequence[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Run prediction for a single image and save visualization.

    Args:
        image_path: Path to input image
        checkpoint_path: Path to .ckpt file. If None, latest from models/ is used
        image_size: Target image size (height, width)
        defect_classes: List of defect class names
        output_path: Where to save visualization.
            Defaults to `visualizations/<stem>_pred.png`

    Returns:
        Path to saved visualization

    Raises:
        FileNotFoundError: If no checkpoint found
    """
    image_path = Path(image_path)

    # Resolve checkpoint
    if checkpoint_path is None:
        models_dir = Path("models")
        ckpts = sorted(
            models_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True
        )
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
    pred_color = mask_to_color(pred_mask, palette)
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
        # Save to visualizations/ instead of next to the source image
        vis_dir = Path("visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_path = vis_dir / f"{image_path.stem}_pred.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return output_path

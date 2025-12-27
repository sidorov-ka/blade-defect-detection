"""Inference module for blade defect detection."""

from blade_defect_detection.inference.predict import mask_to_color, predict_image

__all__ = ["predict_image", "mask_to_color"]

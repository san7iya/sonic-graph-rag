"""Validate album-art files before CLIP sees them."""

from pathlib import Path

import numpy as np
from PIL import Image

MIN_SIDE_PX = 100
MIN_PIXEL_STD = 5.0  # a flat colour/placeholder image has ~0 spread


def validate_image(path: Path) -> list[str]:
    """Return a list of problems (empty list means the image is usable).

    Why these checks: a truncated download or an HTML error page saved as
    .jpg would crash the CLIP batch run mid-way; a tiny or single-colour
    image decodes fine but yields an uninformative vector that would
    silently pollute the image block of the fused embedding.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # detects truncated/corrupt files without a full decode
        with Image.open(path) as img:
            rgb = img.convert("RGB")
    except Exception as e:  # PIL raises many different types for bad files
        return [f"unreadable image ({type(e).__name__})"]

    issues = []
    width, height = rgb.size
    if min(width, height) < MIN_SIDE_PX:
        issues.append(f"image too small ({width}x{height})")
    if float(np.asarray(rgb).std()) < MIN_PIXEL_STD:
        issues.append("image is (nearly) a single flat colour")
    return issues

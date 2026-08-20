"""Wraps CLIP to turn album cover art into vectors.

Pipeline: load image -> CLIP's own preprocessing (resize/crop/normalize
pixel values) -> vision transformer forward pass -> L2-normalize.

CLIP's image and text towers are trained jointly with a contrastive
objective (matching image/caption pairs pulled together, mismatched pairs
pushed apart), which is why a CLIP image vector and a CLIP text vector
live in a comparable space — not why we're comparing CLIP vectors to
MiniLM vectors directly (we're not; fusion concatenates each modality's
own normalized vector instead of asking them to already be aligned to
each other, see fusion.py).
"""

from src import config  # noqa: E402 -- must set HF_HOME before HF libs import

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None


def _get_model_and_processor() -> tuple[CLIPModel, CLIPProcessor]:
    """Lazily load CLIP once per process; loading reads weights from disk
    and is slow enough to avoid repeating per call.
    """
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(config.IMAGE_MODEL_NAME)
        _model.eval()
        _processor = CLIPProcessor.from_pretrained(config.IMAGE_MODEL_NAME)
    return _model, _processor


def encode_image(image_path: Path) -> np.ndarray:
    """Encode one image file into a 512-dim, L2-normalized vector."""
    model, processor = _get_model_and_processor()
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        # In this transformers version, get_image_features() returns a
        # BaseModelOutputWithPooling, not a plain tensor as older docs
        # show; .pooler_output is the 512-dim projected embedding
        # (last_hidden_state is the pre-pooling per-patch tokens).
        features = model.get_image_features(**inputs).pooler_output

    vec = features[0].numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_text_for_image_space(text: str) -> np.ndarray:
    """Encode text using CLIP's *text* tower, landing in the same space
    as encode_image()'s output.

    This is what lets a retrieval query's image sub-vector be non-zero:
    a user query like "moody album art" gets compared against real cover
    art via CLIP's cross-modal alignment, not against other text. Used by
    the retriever (Milestone 4), not by ingestion.
    """
    model, processor = _get_model_and_processor()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output

    vec = features[0].numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


if __name__ == "__main__":
    sample_path = config.RAW_IMAGE_DIR / "queen_bohemian_rhapsody.jpg"
    vec = encode_image(sample_path)
    print("vector shape:", vec.shape)
    print("first 8 values:", np.round(vec[:8], 4))
    print("L2 norm (should be ~1.0):", round(float(np.linalg.norm(vec)), 6))

"""Wraps CLAP (transformers) to turn audio clips into vectors.

Pipeline: load audio -> resample to 48kHz mono (what CLAP was trained on)
-> CLAP's feature extractor (log-mel spectrogram) -> audio transformer
forward pass -> L2-normalize.

Like MiniLM and CLIP, CLAP is trained with a contrastive objective
(matching audio/caption pairs pulled together, mismatched pairs pushed
apart), which is why cosine similarity between its output vectors tracks
perceptual/semantic similarity in sound rather than raw waveform distance.
"""

from src import config  # noqa: E402 -- must set HF_HOME before HF libs import

import os
import shutil
from pathlib import Path

# librosa's m4a/AAC decoding goes through audioread's ffmpeg backend, which
# looks up a binary literally named `ffmpeg`/`ffmpeg.exe` on PATH. No
# ffmpeg is installed on this machine, and imageio-ffmpeg's bundled
# portable binary has a version-suffixed filename (e.g.
# ffmpeg-win-x86_64-v7.1.exe) that PATH lookup by name won't match, so we
# copy it once to a correctly-named file and put that directory on PATH
# before librosa ever tries to load a file.
import imageio_ffmpeg

_FFMPEG_BIN_DIR = config.CACHE_DIR / "bin"
_ffmpeg_dest = _FFMPEG_BIN_DIR / "ffmpeg.exe"
if not _ffmpeg_dest.exists():
    _FFMPEG_BIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), _ffmpeg_dest)
os.environ["PATH"] = str(_FFMPEG_BIN_DIR) + os.pathsep + os.environ["PATH"]

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

CLAP_SAMPLE_RATE = 48_000

_model: ClapModel | None = None
_processor: ClapProcessor | None = None


def _get_model_and_processor() -> tuple[ClapModel, ClapProcessor]:
    """Lazily load CLAP once per process; loading reads weights from disk
    and is slow enough to avoid repeating per call.
    """
    global _model, _processor
    if _model is None:
        _model = ClapModel.from_pretrained(config.AUDIO_MODEL_NAME)
        _model.eval()
        _processor = ClapProcessor.from_pretrained(config.AUDIO_MODEL_NAME)
    return _model, _processor


def encode_audio(audio_path: Path) -> np.ndarray:
    """Encode one audio file into a 512-dim, L2-normalized vector."""
    model, processor = _get_model_and_processor()
    waveform, _ = librosa.load(audio_path, sr=CLAP_SAMPLE_RATE, mono=True)

    inputs = processor(audio=waveform, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        # See image_encoder.py: this transformers version wraps the
        # projected embedding in .pooler_output rather than returning it
        # as a plain tensor.
        features = model.get_audio_features(**inputs).pooler_output

    vec = features[0].numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_text_for_audio_space(text: str) -> np.ndarray:
    """Encode text using CLAP's *text* tower, landing in the same space
    as encode_audio()'s output.

    This is what lets a retrieval query's audio sub-vector be non-zero:
    a query like "sounds like a slow ballad" is compared against real
    audio clips via CLAP's cross-modal alignment, not against other text.
    Used by the retriever (Milestone 4), not by ingestion.
    """
    model, processor = _get_model_and_processor()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output

    vec = features[0].numpy()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


if __name__ == "__main__":
    sample_path = config.RAW_AUDIO_DIR / "queen_bohemian_rhapsody.m4a"
    vec = encode_audio(sample_path)
    print("vector shape:", vec.shape)
    print("first 8 values:", np.round(vec[:8], 4))
    print("L2 norm (should be ~1.0):", round(float(np.linalg.norm(vec)), 6))

"""Validate audio previews before CLAP sees them.

Validation only -- no resampled copies are written. CLAP's own encoder
resamples to 48 kHz (see src/embeddings/audio_encoder.py), so a stored
16 kHz copy would just be a lower-quality input for it. The 16 kHz mono
decode below exists purely to measure duration and loudness cheaply.
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

CHECK_SAMPLE_RATE = 16_000
MIN_SECONDS = 20.0   # iTunes previews are ~30 s; much shorter means a cut-off download
MIN_RMS = 1e-3       # effectively silent below this


def decode_mono_16k(path: Path) -> np.ndarray:
    """Decode a file to a float32 mono 16 kHz array via the bundled ffmpeg.

    Uses imageio-ffmpeg's binary directly instead of importing the CLAP
    module, so validating audio does not pull in torch/transformers.
    Raises RuntimeError if ffmpeg cannot decode the file.
    """
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(CHECK_SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=60)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode(errors="replace")[:200] or "no audio decoded")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def validate_audio(path: Path) -> tuple[list[str], float]:
    """Return (problems, duration_seconds); no problems means usable."""
    try:
        samples = decode_mono_16k(path)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        return [f"undecodable audio ({e})"], 0.0

    duration = len(samples) / CHECK_SAMPLE_RATE
    issues = []
    if duration < MIN_SECONDS:
        issues.append(f"audio too short ({duration:.1f}s < {MIN_SECONDS}s)")
    if float(np.sqrt(np.mean(samples ** 2))) < MIN_RMS:
        issues.append("audio is (nearly) silent")
    return issues, duration

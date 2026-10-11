"""Validators must accept good files and flag each kind of bad file,
using tiny synthetic inputs so the tests need no downloaded data.
"""

import wave

import numpy as np
from PIL import Image

from src.preprocessing.clean_audio import validate_audio
from src.preprocessing.clean_images import validate_image
from src.preprocessing.clean_text import clean_bio, validate_text


def write_wav(path, seconds, amplitude, sr=16_000):
    t = np.arange(int(seconds * sr)) / sr
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def test_clean_bio_strips_citations_headings_and_blank_runs():
    raw = "Song is great.[1] Very good.[citation needed]\n\n\n\n== History ==\nIt charted.  twice."
    cleaned = clean_bio(raw)
    assert "[1]" not in cleaned and "citation needed" not in cleaned and "== History" not in cleaned
    assert "\n\n\n" not in cleaned and "  " not in cleaned
    assert "Song is great." in cleaned and "It charted." in cleaned


def test_validate_text_flags_short_bio():
    assert validate_text("x" * 500) == []
    assert any("too short" in p for p in validate_text("short"))


def test_validate_image_accepts_textured_image(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "ok.jpg"
    Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(path)
    assert validate_image(path) == []


def test_validate_image_flags_flat_small_and_corrupt(tmp_path):
    flat = tmp_path / "flat.jpg"
    Image.new("RGB", (200, 200), (10, 10, 10)).save(flat)
    assert any("flat" in p for p in validate_image(flat))

    small = tmp_path / "small.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (50, 50, 3), dtype=np.uint8)).save(small)
    assert any("too small" in p for p in validate_image(small))

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"<html>404 not found</html>")
    assert any("unreadable" in p for p in validate_image(bad))


def test_validate_audio_accepts_good_clip(tmp_path):
    path = tmp_path / "ok.wav"
    write_wav(path, seconds=25, amplitude=0.5)
    issues, duration = validate_audio(path)
    assert issues == []
    assert abs(duration - 25) < 0.5


def test_validate_audio_flags_short_silent_and_undecodable(tmp_path):
    short = tmp_path / "short.wav"
    write_wav(short, seconds=3, amplitude=0.5)
    assert any("too short" in p for p in validate_audio(short)[0])

    silent = tmp_path / "silent.wav"
    write_wav(silent, seconds=25, amplitude=0.0)
    assert any("silent" in p for p in validate_audio(silent)[0])

    garbage = tmp_path / "garbage.m4a"
    garbage.write_bytes(b"not audio at all")
    assert any("undecodable" in p for p in validate_audio(garbage)[0])

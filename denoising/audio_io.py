"""Audio file loading shared by the command-line scripts."""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file, falling back to ffmpeg for video and compressed formats."""
    try:
        return sf.read(path, dtype="float32")
    except sf.LibsndfileError:
        pass
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    if not probe.stdout.strip():
        raise ValueError(f"{path} has no audio track")
    sr = int(probe.stdout.strip())
    # Decode the first audio stream to mono float32 PCM on stdout.
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "a:0", "-ac", "1", "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(decoded.stdout, dtype=np.float32), sr

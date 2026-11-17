"""Audio file loading and mixing shared by the command-line scripts."""

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
    # frombuffer returns a read-only view; copy so callers (and torch) can write to it.
    return np.frombuffer(decoded.stdout, dtype=np.float32).copy(), sr


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` scaled to the given SNR, then peak-normalize.

    Noise shorter than the speech is looped; longer noise is trimmed. Energy is
    measured over the whole clip, pauses included, which matches what Squim's
    SI-SDR estimate measures.
    """
    noise = np.resize(noise, len(speech))  # np.resize repeats the array to fill
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixture = speech + scale * noise
    # Scaling speech and noise together leaves the SNR unchanged.
    return (mixture * 0.9 / np.max(np.abs(mixture))).astype(np.float32)

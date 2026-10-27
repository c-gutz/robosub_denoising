"""Common interface shared by every denoising backend."""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torchaudio.functional as AF


class Denoiser(ABC):
    """Wraps a speech enhancement model behind a numpy-in / numpy-out API.

    Subclasses set ``name`` and ``native_sr`` and implement ``_enhance``,
    which receives mono float32 audio already resampled to ``native_sr``.
    Resampling, channel mixing and dtype handling live here so every model
    is called the same way by the benchmark and the live pipeline.
    """

    name: str
    native_sr: int

    @abstractmethod
    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        """Denoise mono float32 audio sampled at ``native_sr``."""

    def denoise(self, audio: np.ndarray, sr: int, out_sr: int | None = None) -> np.ndarray:
        """Denoise ``audio`` sampled at ``sr``.

        Args:
            audio: Shape (samples,) or (samples, channels). Integer PCM is
                scaled to [-1, 1]; multichannel input is averaged to mono.
            sr: Sample rate of ``audio``.
            out_sr: Sample rate of the result. Defaults to ``sr``. Pass 16000
                to feed Whisper directly.

        Returns:
            Mono float32 array at ``out_sr``.
        """
        out_sr = out_sr or sr
        audio = to_mono_float(audio)
        audio = resample(audio, sr, self.native_sr)
        enhanced = self._enhance(audio)
        return resample(enhanced, self.native_sr, out_sr)


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def resample(audio: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return audio
    out = AF.resample(torch.from_numpy(audio), orig_sr, new_sr)
    return out.numpy()

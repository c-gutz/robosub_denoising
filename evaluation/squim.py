"""TorchAudio-Squim: estimate SNR, intelligibility and quality without a clean reference.

Squim (Kumar et al., ICASSP 2023) predicts, from the noisy clip alone:
    snr   SI-SDR in dB, roughly the signal-to-noise ratio (higher is better;
          0 dB = speech and noise equally loud, negative = noise louder)
    stoi  speech intelligibility, 0-1 (higher is better)
    pesq  perceptual quality, 1-4.5 (higher is better)

Weights (~28 MB) download on first use. Inputs are resampled to 16 kHz.
"""

from functools import cache

import numpy as np
import torch

from denoising.base import resample, to_mono_float

SQUIM_SR = 16_000


@cache
def _model():
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    return SQUIM_OBJECTIVE.get_model().eval()


def squim(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Score audio and return estimated SNR (SI-SDR, dB), STOI and PESQ."""
    audio = resample(to_mono_float(audio), sr, SQUIM_SR)
    with torch.inference_mode():
        stoi, pesq, sisdr = _model()(torch.from_numpy(audio)[None])
    return {"snr": sisdr.item(), "stoi": stoi.item(), "pesq": pesq.item()}

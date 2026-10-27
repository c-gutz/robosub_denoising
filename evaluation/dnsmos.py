"""DNSMOS: reference-free speech quality scores (1-5, higher is better).

Uses Microsoft's DNSMOS P.835 and P.808 models via torchmetrics, which
downloads the ONNX weights on first use. Inputs are resampled to 16 kHz.
"""

import numpy as np
import torch

from denoising.base import resample, to_mono_float

DNSMOS_SR = 16_000

def dnsmos(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Score audio and return P808, SIG, BAK and OVRL MOS estimates."""
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    audio = resample(to_mono_float(audio), sr, DNSMOS_SR)
    scores = deep_noise_suppression_mean_opinion_score(
        torch.from_numpy(audio), fs=DNSMOS_SR, personalized=False
    )

    p808, sig, bak, ovrl = scores.tolist()
    return {"p808": p808, "sig": sig, "bak": bak, "ovrl": ovrl}
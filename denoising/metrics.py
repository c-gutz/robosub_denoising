"""Speech quality metrics."""

import numpy as np

from .base import resample, to_mono_float

DNSMOS_SR = 16_000


def dnsmos(audio: np.ndarray, sr: int, personalized: bool = False) -> dict[str, float]:
    """Score speech quality with Microsoft's DNSMOS (no clean reference needed).

    Scores range 1-5, higher is better:
        SIG:  speech signal quality (distortion of the voice itself)
        BAK:  background noise intrusiveness (5 = no audible noise)
        OVRL: overall quality (ITU-T P.835)
        P808: overall quality (ITU-T P.808 model)

    Clips are scored in 9 s windows with a 1 s hop and averaged; clips shorter
    than 9 s are looped to fill a window.

    Args:
        audio: Shape (samples,) or (samples, channels), any sample rate.
        sr: Sample rate of ``audio``.
        personalized: Use the personalized DNSMOS model, which also counts
            interfering background talkers as noise (useful for babble, e.g.
            a crowded dining hall).
    """
    # Imported here so the rest of the package doesn't require onnxruntime.
    from speechmos import dnsmos as _dnsmos

    audio = resample(to_mono_float(audio), sr, DNSMOS_SR)
    # Resampling and enhancement can overshoot slightly; DNSMOS rejects |x| > 1.
    audio = np.clip(audio, -1.0, 1.0)
    scores = _dnsmos.run(audio, DNSMOS_SR, model_type="dnsmos_personalized" if personalized else "dnsmos")
    return {
        "SIG": float(scores["sig_mos"]),
        "BAK": float(scores["bak_mos"]),
        "OVRL": float(scores["ovrl_mos"]),
        "P808": float(scores["p808_mos"]),
    }


def format_dnsmos(scores: dict[str, float]) -> str:
    return "DNSMOS " + "  ".join(f"{k} {v:.2f}" for k, v in scores.items())

"""FRCRN_SE_16K backend, run through ClearerVoice-Studio.

Runs at 16 kHz, Whisper's input rate, so no 48 kHz round trip is needed.
Weights: https://huggingface.co/alibabasglab/FRCRN_SE_16K
"""

import numpy as np
import torch

from .clearvoice import ClearVoiceDenoiser


class FRCRNDenoiser(ClearVoiceDenoiser):
    name = "frcrn_se_16k"
    native_sr = 16_000
    clearvoice_name = "FRCRN_SE_16K"

    @torch.no_grad()
    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        from clearvoice.dataloader.dataloader import audio_norm

        # ClearerVoice's file pipeline normalizes loudness to -25 dB before the
        # model and undoes it after; its array API skips this, so do it here.
        normed, scale = audio_norm(audio.astype(np.float64))
        out = self.network.decode_data(normed[np.newaxis, :])
        # decode_data pads the input to its window size and doesn't trim it back.
        return (np.squeeze(out)[: len(audio)] * scale).astype(np.float32)

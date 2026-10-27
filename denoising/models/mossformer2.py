"""MossFormer2_SE_48K backend, run through ClearerVoice-Studio.

Runs at 48 kHz. Heavier than the other models, and likely the highest-quality
denoiser on the list, so it serves as an upper bound on what denoising can do.
Weights: https://huggingface.co/alibabasglab/MossFormer2_SE_48K
"""

import numpy as np
import torch

from .clearvoice import ClearVoiceDenoiser


class MossFormer2Denoiser(ClearVoiceDenoiser):
    name = "mossformer2_se_48k"
    native_sr = 48_000
    clearvoice_name = "MossFormer2_SE_48K"

    @torch.no_grad()
    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        # Use ClearerVoice's file-pipeline decoder rather than its array API
        # (decode_data): the array version never splits clips longer than 20 s
        # into windows (it checks the batch dim instead of length), and its
        # windowed path would crash anyway. This one windows long clips with
        # 4 s overlapping segments. Unlike FRCRN, ClearerVoice doesn't
        # loudness-normalize MossFormer2 input.
        from clearvoice.utils.decode import decode_one_audio

        out = decode_one_audio(self.network.model, self.network.device, audio[np.newaxis, :], self.network.args)
        out = out.numpy() if isinstance(out, torch.Tensor) else np.asarray(out)
        return np.squeeze(out)[: len(audio)].astype(np.float32)

"""FRCRN_SE_16K backend, run through ClearerVoice-Studio.

Runs at 16 kHz, Whisper's input rate, so no 48 kHz round trip is needed.
Weights: https://huggingface.co/alibabasglab/FRCRN_SE_16K
"""

import types

import numpy as np
import torch
import torch.nn.functional as F

from .clearvoice import ClearVoiceDenoiser


class FRCRNDenoiser(ClearVoiceDenoiser):
    name = "frcrn_se_16k"
    native_sr = 16_000
    clearvoice_name = "FRCRN_SE_16K"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        speed_up_depthwise_convs(self.network.model)

    @torch.no_grad()
    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        from clearvoice.dataloader.dataloader import audio_norm

        if np.ptp(audio) == 0:  # digital silence: audio_norm would return NaN
            return np.zeros_like(audio, dtype=np.float32)
        # ClearerVoice's file pipeline normalizes loudness to -25 dB before the
        # model and undoes it after; its array API skips this, so do it here.
        normed, scale = audio_norm(audio.astype(np.float64))
        out = self.network.decode_data(normed[np.newaxis, :])
        # decode_data pads the input to its window size and doesn't trim it back.
        return (np.squeeze(out)[: len(audio)] * scale).astype(np.float32)


def speed_up_depthwise_convs(model: torch.nn.Module) -> int:
    """Swap FRCRN's depthwise (k x 1) Conv2d layers for an unfold + einsum.

    FRCRN's FSMN memory blocks use Conv2d(128, 128, (20, 1), groups=128). PyTorch
    builds without oneDNN (e.g. macOS arm64) run that as a Python-level loop over
    every group and batch item, ~14k tiny convs per second of audio, which was
    ~70% of FRCRN's CPU time. The replacement computes the same thing (max
    difference ~1e-7) about 2x faster end to end. Returns the number of layers patched.
    """
    patched = 0
    for module in model.modules():
        if (isinstance(module, torch.nn.Conv2d) and module.kernel_size[1] == 1
                and module.groups == module.in_channels == module.out_channels > 1
                and module.padding[1] == 0 and module.stride[1] == 1 and module.padding_mode == "zeros"):
            module.forward = types.MethodType(_depthwise_kx1_forward, module)
            patched += 1
    return patched


def _depthwise_kx1_forward(self: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    # x: (batch, channels, length, 1)
    x = x.squeeze(-1)
    if self.padding[0]:
        x = F.pad(x, (self.padding[0], self.padding[0]))
    k, dilation = self.kernel_size[0], self.dilation[0]
    windows = x.unfold(2, (k - 1) * dilation + 1, self.stride[0])[..., ::dilation]  # (B, C, L', k)
    y = torch.einsum("bclk,ck->bcl", windows, self.weight[:, 0, :, 0])
    if self.bias is not None:
        y = y + self.bias[:, None]
    return y.unsqueeze(-1)

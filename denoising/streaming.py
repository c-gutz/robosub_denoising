"""Block-by-block FRCRN for live audio.

FRCRN looks at a whole clip at once, so to run it on a live stream we feed it
overlapping windows: each step runs the model on

    [ context (past) | block (new output) | lookahead (future) ]

and keeps only the block's output. The past context and the lookahead give the
model enough surroundings that the block's edges come out like the middle of a
long clip, and a short crossfade (taken from the lookahead) hides what seam is left.

The stream comes out delayed by block + lookahead, plus however long the model takes
on one window. The model must finish a window in under one block of time, or
the stream falls behind.
"""

import numpy as np
import torch

from .models.frcrn import FRCRNDenoiser

FRAME = 320  # FRCRN's STFT hop at 16 kHz (20 ms); window sizes are rounded to it.


class StreamingFRCRN:
    def __init__(self, denoiser: FRCRNDenoiser | None = None, block_s: float = 0.6,
                 context_s: float = 0.4, lookahead_s: float = 0.16, fade_s: float = 0.04):
        """
        Args:
            denoiser: A loaded FRCRNDenoiser (e.g. fine-tuned weights); None loads the pretrained one.
            block_s: New audio produced per model call. Bigger means less work per second
                of audio but more delay.
            context_s: Past audio the model sees before each block.
            lookahead_s: Future audio the model sees after each block. Adds directly to the delay.
            fade_s: Crossfade between consecutive blocks; at most ``lookahead_s``.
        """
        self.denoiser = denoiser or FRCRNDenoiser()
        self.sr = self.denoiser.native_sr
        self.block = _frames(block_s, self.sr)
        self.context = _frames(context_s, self.sr)
        self.lookahead = _frames(lookahead_s, self.sr)
        self.fade = min(int(fade_s * self.sr), self.lookahead)
        if self.block <= 0:
            raise ValueError("block_s must be at least one 20 ms frame")
        # Equal-power fades, so a seam between two uncorrelated blocks doesn't dip in level.
        t = (np.arange(self.fade) + 0.5) / max(self.fade, 1)
        self._fade_in, self._fade_out = np.sin(t * np.pi / 2), np.cos(t * np.pi / 2)
        self.reset()

    @property
    def window(self) -> int:
        """Samples the model sees per call."""
        return self.context + self.block + self.lookahead

    @property
    def delay_s(self) -> float:
        """Delay from the stream's structure alone, not counting model time."""
        return (self.block + self.lookahead) / self.sr

    def reset(self) -> None:
        self._window = np.zeros(self.window, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._tail = np.zeros(self.fade, dtype=np.float32)

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed any number of new 16 kHz samples; returns the denoised samples now ready.

        The output is the input delayed by ``lookahead`` samples (and it comes out in
        whole blocks), so it lines up with the input once that delay is accounted for.
        """
        self._pending = np.concatenate([self._pending, np.asarray(chunk, dtype=np.float32)])
        out = []
        while len(self._pending) >= self.block:
            new, self._pending = self._pending[: self.block], self._pending[self.block:]
            out.append(self._step(new))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """Push silence through so the last real samples come out."""
        n = len(self._pending) + self.lookahead
        padded = -(-n // self.block) * self.block  # round up to whole blocks
        out = self.process(np.zeros(padded - len(self._pending), dtype=np.float32))
        return out[:n]

    @torch.no_grad()
    def _step(self, new: np.ndarray) -> np.ndarray:
        from clearvoice.dataloader.dataloader import audio_norm

        self._window = np.concatenate([self._window[self.block:], new])
        # Same loudness normalization as the whole-clip path, computed per window and
        # undone afterwards. (Holding the level from recent loud windows instead, so
        # noise-only windows aren't lifted to speech level, made no difference on
        # recordings 1-4.)
        if np.ptp(self._window) == 0:
            # Digital silence (muted mic, stream start): audio_norm would return NaN.
            enhanced = np.zeros(self.window)
        else:
            normed, scale = audio_norm(self._window.astype(np.float64))
            model = self.denoiser.network.model
            x = torch.from_numpy(normed.astype(np.float32))[None].to(self.denoiser.network.device)
            enhanced = model.inference(x).cpu().numpy().reshape(-1)[: self.window] * scale
            enhanced = np.nan_to_num(enhanced, nan=0.0, posinf=0.0, neginf=0.0)

        start = self.context
        out = enhanced[start: start + self.block].astype(np.float32)
        if self.fade:
            out[: self.fade] = out[: self.fade] * self._fade_in + self._tail * self._fade_out
            self._tail = enhanced[start + self.block: start + self.block + self.fade].astype(np.float32)
        return out


def _frames(seconds: float, sr: int) -> int:
    return int(round(seconds * sr / FRAME)) * FRAME

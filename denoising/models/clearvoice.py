"""Shared loading for models run through ClearerVoice-Studio (pip install clearvoice).

Pretrained weights are downloaded from https://huggingface.co/alibabasglab/<model>
on first use to ~/.cache/clearvoice/<model>.
"""

import importlib
from pathlib import Path

import torch

from ..base import Denoiser

CACHE_DIR = Path.home() / ".cache" / "clearvoice"


class ClearVoiceDenoiser(Denoiser):
    """Base for ClearerVoice speech enhancement models.

    Subclasses set ``clearvoice_name`` (e.g. "FRCRN_SE_16K"), which selects
    both the ClearerVoice config and its ``clearvoice.networks.CLS_<name>``
    class, and implement ``_enhance``.
    """

    clearvoice_name: str

    def __init__(self, checkpoint_dir: str | None = None, device: str | None = None):
        """
        Args:
            checkpoint_dir: Directory holding a ClearerVoice checkpoint (with a
                ``last_best_checkpoint`` file), e.g. a fine-tuned one. ``None``
                uses the pretrained weights, downloading them if needed.
            device: "cuda", "mps" or "cpu". ``None`` uses CUDA if available,
                otherwise CPU (FRCRN measured ~2x slower on Apple MPS than CPU).
        """
        # Imported here so other backends don't require clearvoice.
        from clearvoice.network_wrapper import network_wrapper

        # ClearVoice() hardcodes a checkpoint dir relative to the current
        # directory, so build the args ourselves to point it at the cache.
        wrapper = network_wrapper()
        wrapper.model_name = self.clearvoice_name
        wrapper.load_args_se()
        args = wrapper.args
        args.task = "speech_enhancement"
        args.network = self.clearvoice_name
        args.checkpoint_dir = str(checkpoint_dir or CACHE_DIR / self.clearvoice_name)

        network_cls = getattr(importlib.import_module("clearvoice.networks"), f"CLS_{self.clearvoice_name}")
        self.network = network_cls(args)
        if not (Path(args.checkpoint_dir) / "last_best_checkpoint").is_file():
            raise RuntimeError(f"No {self.clearvoice_name} checkpoint found in {args.checkpoint_dir}")

        # ClearerVoice picks MPS over CPU on its own; override that default.
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.device = device
        self.network.model.to(device)

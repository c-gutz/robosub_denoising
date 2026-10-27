"""DeepFilterNet3 backend (https://github.com/Rikorose/DeepFilterNet).

Runs at 48 kHz. Pretrained weights are downloaded automatically on first use
to the DeepFilterNet cache directory (~/.cache/DeepFilterNet on Linux).
"""

import numpy as np
import torch

from ..base import Denoiser


class DeepFilterNetDenoiser(Denoiser):
    name = "deepfilternet3"
    native_sr = 48_000

    def __init__(
        self,
        model_dir: str | None = None,
        post_filter: bool = False,
        atten_lim_db: float | None = None,
    ):
        """
        Args:
            model_dir: Path to a model/checkpoint directory, e.g. a fine-tuned
                checkpoint. ``None`` downloads the pretrained DeepFilterNet3.
            post_filter: Enable DeepFilterNet's post filter, which attenuates
                very noisy sections further at some cost to speech quality.
            atten_lim_db: Cap noise attenuation in dB. ``None`` removes as
                much noise as possible; a limit (e.g. 12) can sound more
                natural and sometimes helps ASR.
        """
        # Imported here so other backends don't require deepfilternet.
        from df.enhance import init_df

        self.model, self.df_state, _ = init_df(
            model_base_dir=model_dir,
            post_filter=post_filter,
            log_level="WARNING",
            log_file=None,
        )
        self.model.eval()
        self.atten_lim_db = atten_lim_db
        # init_df only reports the model's sample rate through df_state.
        assert self.df_state.sr() == self.native_sr

    @torch.no_grad()
    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        from df.enhance import enhance

        out = enhance(
            self.model,
            self.df_state,
            torch.from_numpy(audio).unsqueeze(0),
            atten_lim_db=self.atten_lim_db,
        )
        return out.squeeze(0).cpu().numpy()

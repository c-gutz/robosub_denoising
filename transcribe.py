"""Transcribe speech from noisy recordings with Whisper.

By default the raw audio goes straight to Whisper, which is already fairly
robust to noise. --denoise runs a denoiser (DeepFilterNet3 by default) first;
--compare prints both transcripts to check whether denoising actually helps,
since enhancement artifacts can also make recognition worse.

Accepts anything ffmpeg can decode, including iPhone videos (.mov/.mp4) and
voice memos (.m4a).

Examples:
    python transcribe.py IMG_0123.MOV
    python transcribe.py IMG_0123.MOV --denoise
    python transcribe.py IMG_0123.MOV --compare
    python transcribe.py clips/*.wav --whisper openai/whisper-small.en
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import pipeline

from denoising import DENOISERS, load_denoiser
from denoising.audio_io import load_audio
from denoising.base import resample, to_mono_float

WHISPER_SR = 16_000
WHISPER_WINDOW_S = 30


def load_whisper(model_id: str):
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        dtype=dtype,
    )


def transcribe(asr, audio_16k: np.ndarray, language: str | None) -> str:
    generate_kwargs = {"task": "transcribe"}
    # Clips longer than Whisper's 30 s window need timestamps for long-form decoding.
    if len(audio_16k) > WHISPER_WINDOW_S * WHISPER_SR:
        generate_kwargs["return_timestamps"] = True
    # English-only (.en) checkpoints reject the language argument.
    if language and not asr.model.name_or_path.endswith(".en"):
        generate_kwargs["language"] = language
    out = asr({"raw": audio_16k, "sampling_rate": WHISPER_SR}, generate_kwargs=generate_kwargs)
    return out["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio or video files")
    parser.add_argument("-m", "--model", default="deepfilternet3", choices=sorted(DENOISERS), help="Denoiser for --denoise/--compare")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--denoise", action="store_true", help="Denoise before transcribing")
    mode.add_argument("--compare", action="store_true", help="Show raw and denoised transcripts side by side")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    asr = load_whisper(args.whisper)
    denoiser = load_denoiser(args.model) if args.denoise or args.compare else None

    for path in args.inputs:
        audio, sr = load_audio(path)
        audio = to_mono_float(audio)
        print(f"== {path}")

        if denoiser is None or args.compare:
            start = time.perf_counter()
            text = transcribe(asr, resample(audio, sr, WHISPER_SR), args.language)
            print(f"[raw {time.perf_counter() - start:.1f}s] {text}")

        if denoiser is not None:
            start = time.perf_counter()
            clean = denoiser.denoise(audio, sr, out_sr=WHISPER_SR)
            text = transcribe(asr, clean, args.language)
            print(f"[{denoiser.name} {time.perf_counter() - start:.1f}s] {text}")


if __name__ == "__main__":
    main()

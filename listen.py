"""Write denoised audio you can listen to, plus before/after transcripts.

For each recording, runs every denoiser and saves, in outputs/<clip>/:
    raw.wav                 the original audio track (for A/B listening)
    <denoiser>.wav          one file per denoiser, at the recording's sample rate
    transcripts.txt         Whisper's transcript of raw and of each denoised version,
                            plus the reference and WER if <clip>.txt exists

Transcripts are also printed to the terminal.

Examples:
    python listen.py recordings/*/*.MOV
    python listen.py IMG_0123.MOV -m deepfilternet3 frcrn_se_16k
    python listen.py IMG_0123.MOV -o my_outputs --whisper openai/whisper-large-v3-turbo
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import transformers

from denoising import DENOISERS, load_denoiser
from denoising.audio_io import load_audio
from denoising.base import resample, to_mono_float
from evaluation.wer import wer
from transcribe import WHISPER_SR, load_whisper, transcribe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio or video files")
    parser.add_argument("-m", "--models", nargs="+", default=sorted(DENOISERS), choices=sorted(DENOISERS),
                        help="Denoisers to run (default: all)")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    asr = load_whisper(args.whisper)
    # None stands for the raw, undenoised audio.
    systems = {"raw": None} | {name: load_denoiser(name) for name in args.models}

    for path in args.inputs:
        audio, sr = load_audio(path)
        audio = to_mono_float(audio)
        ref_path = path.with_suffix(".txt")
        reference = ref_path.read_text().strip() if ref_path.exists() else None

        clip_dir = args.out_dir / path.stem
        clip_dir.mkdir(parents=True, exist_ok=True)
        lines = [f"== {path}"]
        if reference is not None:
            lines.append(f"[reference] {' '.join(reference.split())}")
        print(lines[0])

        for name, denoiser in systems.items():
            out = audio if denoiser is None else denoiser.denoise(audio, sr)
            # Clip so loud enhancement output doesn't wrap around in 16-bit WAV.
            sf.write(clip_dir / f"{name}.wav", np.clip(out, -1.0, 1.0), sr)

            text = transcribe(asr, resample(out, sr, WHISPER_SR), args.language)
            score = ""
            if reference is not None:
                score = f" (WER {wer(reference, text, normalize=asr.tokenizer.normalize)['wer']:.1%})"
            line = f"[{name}{score}] {text}"
            lines.append(line)
            print(line)

        (clip_dir / "transcripts.txt").write_text("\n\n".join(lines) + "\n")
        print(f"   -> {clip_dir}/\n")


if __name__ == "__main__":
    main()

"""Denoise audio files from the command line.

Accepts anything ffmpeg can decode, including iPhone videos (.mov/.mp4) and
voice memos (.m4a); the audio track is extracted automatically.

Examples:
    python denoise_file.py noisy.wav
    python denoise_file.py IMG_0123.MOV
    python denoise_file.py recordings/*.wav -o denoised/ --out-sr 16000
"""

import argparse
import time
from pathlib import Path

import soundfile as sf

from denoising import DENOISERS, load_denoiser
from denoising.audio_io import load_audio
from denoising.metrics import dnsmos, format_dnsmos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files to denoise")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("denoised"))
    parser.add_argument("-m", "--model", default="deepfilternet3", choices=sorted(DENOISERS))
    parser.add_argument("--out-sr", type=int, default=None, help="Output sample rate (default: same as input)")
    parser.add_argument("--dnsmos", action="store_true", help="Print DNSMOS scores before and after denoising")
    parser.add_argument("--personalized", action="store_true", help="Use personalized DNSMOS (penalizes background talkers)")
    args = parser.parse_args()

    denoiser = load_denoiser(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        audio, sr = load_audio(path)
        out_sr = args.out_sr or sr

        start = time.perf_counter()
        enhanced = denoiser.denoise(audio, sr, out_sr=out_sr)
        elapsed = time.perf_counter() - start

        out_path = args.out_dir / f"{path.stem}_{denoiser.name}.wav"
        sf.write(out_path, enhanced, out_sr)
        rtf = elapsed / (len(audio) / sr)
        print(f"{path} -> {out_path}  ({elapsed:.2f}s, RTF {rtf:.3f})")
        if args.dnsmos:
            print(f"    before: {format_dnsmos(dnsmos(audio, sr, args.personalized))}")
            print(f"    after:  {format_dnsmos(dnsmos(enhanced, out_sr, args.personalized))}")


if __name__ == "__main__":
    main()

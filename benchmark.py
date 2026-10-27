"""Benchmark denoisers on recordings: DNSMOS quality, Whisper WER and speed.

For each clip, scores the raw audio plus each --models denoiser. If a
transcript sits next to the clip (beach.MOV -> beach.txt), also reports
Whisper's word error rate against it.

Examples:
    python benchmark.py clips/*.MOV                     # raw vs every denoiser
    python benchmark.py clips/* -m deepfilternet3 --csv results.csv
"""

import argparse
import csv
import time
from pathlib import Path

import transformers

from denoising import DENOISERS, load_denoiser
from denoising.audio_io import load_audio
from denoising.base import resample, to_mono_float
from evaluation.dnsmos import dnsmos
from evaluation.wer import wer
from transcribe import WHISPER_SR, load_whisper, transcribe

COLUMNS = ["sig", "bak", "ovrl", "wer", "rtf"]
CSV_FIELDS = ["clip", "system", "p808", "sig", "bak", "ovrl", "wer",
              "substitutions", "deletions", "insertions", "rtf", "text", "reference"]


def format_row(row: dict) -> str:
    cells = [f"{row[c]:6.2f}" if row.get(c) is not None else f"{'-':>6}" for c in COLUMNS]
    return f"{row['clip']:<24} {row['system']:<16} " + " ".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio or video files")
    parser.add_argument("-m", "--models", nargs="+", default=sorted(DENOISERS), choices=sorted(DENOISERS),
                        help="Denoisers to compare against raw (default: all)")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    parser.add_argument("--csv", type=Path, help="Also write per-clip results to this CSV")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    asr = load_whisper(args.whisper)
    normalize = asr.tokenizer.normalize
    # None stands for the raw, undenoised baseline.
    systems = {"raw": None} | {name: load_denoiser(name) for name in args.models}

    print(f"{'clip':<24} {'system':<16} " + " ".join(f"{c.upper():>6}" for c in COLUMNS))
    rows = []
    for path in args.inputs:
        audio, sr = load_audio(path)
        audio = to_mono_float(audio)
        ref_path = path.with_suffix(".txt")
        reference = ref_path.read_text().strip() if ref_path.exists() else None

        for name, denoiser in systems.items():
            start = time.perf_counter()
            if denoiser is None:
                audio_16k = resample(audio, sr, WHISPER_SR)
            else:
                audio_16k = denoiser.denoise(audio, sr, out_sr=WHISPER_SR)
            rtf = (time.perf_counter() - start) / (len(audio) / sr)

            text = transcribe(asr, audio_16k, args.language)
            row = {"clip": path.name, "system": name, "rtf": rtf, "text": text,
                   "reference": reference, **dnsmos(audio_16k, WHISPER_SR)}
            if reference is not None:
                row |= wer(reference, text, normalize=normalize)
            rows.append(row)
            print(format_row(row))

    print("\nSummary: mean DNSMOS and RTF; WER pooled over clips with transcripts")
    for name in systems:
        system_rows = [r for r in rows if r["system"] == name]
        summary = {"clip": "ALL", "system": name}
        for c in ("sig", "bak", "ovrl", "rtf"):
            summary[c] = sum(r[c] for r in system_rows) / len(system_rows)
        scored = [r for r in system_rows if r["reference"] is not None]
        if scored:
            summary["wer"] = wer([r["reference"] for r in scored], [r["text"] for r in scored],
                                 normalize=normalize)["wer"]
        print(format_row(summary))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

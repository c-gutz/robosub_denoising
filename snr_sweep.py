"""Mix clean speech with noise at known SNRs and see how everything holds up.

Real recordings never tell you their true SNR. This builds test clips where
you do know it: clean speech + noise, with the noise scaled so that
    SNR (dB) = 10 * log10(speech energy / noise energy)
hits each --snrs value exactly. For every mixture and every system (raw and
each denoiser) it reports:
    est    Squim's estimated SNR, to check against the true SNR
    stoi   Squim's estimated intelligibility (0-1)
    wer    Whisper word error rate

The reference transcript is <clean>.txt if it exists, otherwise Whisper's own
transcript of the clean clip (so WER then measures how much the noise changes
Whisper's output, not absolute accuracy).

Examples:
    python snr_sweep.py quiet_room.wav --noise dining_hall.wav wind.wav
    python snr_sweep.py quiet_room.wav --noise babble.wav --snrs 0 -5 -10 -m frcrn_se_16k
    python snr_sweep.py quiet_room.wav --noise babble.wav --save-dir mixtures --csv sweep.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import transformers

from denoising import DENOISERS, load_denoiser
from denoising.audio_io import load_audio
from denoising.base import resample, to_mono_float
from evaluation.squim import squim
from evaluation.wer import wer
from transcribe import WHISPER_SR, load_whisper, transcribe

DEFAULT_SNRS = [20, 10, 5, 0, -5, -10, -15]
CSV_FIELDS = ["noise", "true_snr", "system", "snr", "stoi", "pesq", "wer", "text"]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``speech`` scaled to the given SNR, then peak-normalize.

    Noise shorter than the speech is looped; longer noise is trimmed. Energy is
    measured over the whole clip, pauses included, which matches what Squim's
    SI-SDR estimate measures.
    """
    noise = np.resize(noise, len(speech))  # np.resize repeats the array to fill
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixture = speech + scale * noise
    # Scaling speech and noise together leaves the SNR unchanged.
    return (mixture * 0.9 / np.max(np.abs(mixture))).astype(np.float32)


def load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, file_sr = load_audio(path)
    audio = to_mono_float(audio)
    if sr is None:
        return audio, file_sr
    return resample(audio, file_sr, sr), sr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clean", type=Path, help="Clean speech recording (quiet room)")
    parser.add_argument("--noise", nargs="+", type=Path, required=True, help="Noise-only recordings")
    parser.add_argument("--snrs", nargs="+", type=float, default=DEFAULT_SNRS, help="True SNRs to test, in dB")
    parser.add_argument("-m", "--models", nargs="+", default=sorted(DENOISERS), choices=sorted(DENOISERS),
                        help="Denoisers to compare against raw (default: all)")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    parser.add_argument("--save-dir", type=Path, help="Also write each mixture as a .wav to listen to")
    parser.add_argument("--csv", type=Path, help="Also write per-mixture results to this CSV")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    asr = load_whisper(args.whisper)
    normalize = asr.tokenizer.normalize
    # None stands for the raw, undenoised mixture.
    systems = {"raw": None} | {name: load_denoiser(name) for name in args.models}

    speech, sr = load_mono(args.clean)
    ref_path = args.clean.with_suffix(".txt")
    if ref_path.exists():
        reference = ref_path.read_text().strip()
        print(f"Reference ({ref_path.name}): {' '.join(reference.split())}")
    else:
        reference = transcribe(asr, resample(speech, sr, WHISPER_SR), args.language)
        print(f"No {ref_path.name}; using Whisper's transcript of the clean clip as reference:\n  {reference}")
    clean_scores = squim(speech, sr)
    print(f"Squim on the clean clip: est SNR {clean_scores['snr']:.1f} dB, STOI {clean_scores['stoi']:.2f}\n")

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    header = f"{'noise':<20} {'true':>6} {'system':<20} {'est':>6} {'stoi':>5} {'wer':>7}"
    print(header)
    rows = []
    for noise_path in args.noise:
        noise, _ = load_mono(noise_path, sr)
        for snr_db in args.snrs:
            mixture = mix_at_snr(speech, noise, snr_db)
            if args.save_dir:
                sf.write(args.save_dir / f"{args.clean.stem}+{noise_path.stem}_{snr_db:+g}dB.wav", mixture, sr)
            for name, denoiser in systems.items():
                out = mixture if denoiser is None else denoiser.denoise(mixture, sr)
                text = transcribe(asr, resample(out, sr, WHISPER_SR), args.language)
                row = {"noise": noise_path.stem, "true_snr": snr_db, "system": name, "text": text,
                       **squim(out, sr), "wer": wer(reference, text, normalize=normalize)["wer"]}
                rows.append(row)
                print(f"{row['noise']:<20} {snr_db:>6g} {name:<20} {row['snr']:>6.1f} {row['stoi']:>5.2f} "
                      f"{row['wer']:>7.1%}")

    print("\nIs Squim's SNR estimate right? (raw mixtures, averaged over noises)")
    print(f"{'true':>6} {'est':>6} {'error':>6}")
    for snr_db in args.snrs:
        est = np.mean([r["snr"] for r in rows if r["system"] == "raw" and r["true_snr"] == snr_db])
        print(f"{snr_db:>6g} {est:>6.1f} {est - snr_db:>+6.1f}")

    print("\nWhisper WER by true SNR (pooled over noises)")
    print(f"{'true':>6} " + " ".join(f"{name:>18}" for name in systems))
    for snr_db in args.snrs:
        cells = []
        for name in systems:
            texts = [r["text"] for r in rows if r["system"] == name and r["true_snr"] == snr_db]
            cells.append(f"{wer([reference] * len(texts), texts, normalize=normalize)['wer']:>18.1%}")
        print(f"{snr_db:>6g} " + " ".join(cells))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

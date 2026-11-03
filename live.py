"""Push-to-talk transcription from the laptop microphone.

Press Enter to start talking, Enter again to stop. Each take is transcribed
raw and after each denoiser, with Squim's estimated SNR of the raw take, and
saved as .wav files so you can listen to what Whisper heard.

Examples:
    python live.py                                  # raw vs frcrn_se_16k
    python live.py -m deepfilternet3 frcrn_se_16k   # compare several denoisers
    python live.py --list-devices                   # find your mic's index
    python live.py --device 1 --no-save
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import transformers

from denoising import DENOISERS, load_denoiser
from denoising.base import resample
from evaluation.squim import squim
from transcribe import WHISPER_SR, load_whisper, transcribe

MIC_SR = 48_000
MIN_TAKE_S = 0.3
PERMISSION_HINT = ("If the mic can't be opened, allow your terminal app under System Settings -> "
                   "Privacy & Security -> Microphone, or pick another --device (see --list-devices).")


def record_take(sd, device) -> np.ndarray:
    """Record from the mic until the user presses Enter."""
    chunks = []
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", device=device,
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("Recording... press Enter to stop.")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--models", nargs="+", default=["frcrn_se_16k"], choices=sorted(DENOISERS),
                        help="Denoisers to compare against raw (default: frcrn_se_16k)")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    parser.add_argument("--device", help="Input device index or name (default: system mic)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--save-dir", type=Path, default=Path("takes"), help="Where to save each take's .wav files")
    parser.add_argument("--no-save", action="store_true", help="Don't save takes")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    print("Loading Whisper and denoisers...")
    transformers.logging.set_verbosity_error()
    asr = load_whisper(args.whisper)
    # None stands for the raw, undenoised take.
    systems = {"raw": None} | {name: load_denoiser(name) for name in args.models}
    # Warm up every model so the first take isn't slowed down by one-time setup.
    silence = np.zeros(MIC_SR, dtype=np.float32)
    for denoiser in systems.values():
        warm = silence if denoiser is None else denoiser.denoise(silence, MIC_SR)
        transcribe(asr, resample(warm, MIC_SR, WHISPER_SR), args.language)
    squim(silence, MIC_SR)

    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    take_num = len(list(args.save_dir.glob("take_*_raw.wav"))) if args.save_dir.exists() else 0

    try:
        while True:
            if input("\nPress Enter to talk (q + Enter to quit): ").strip().lower() == "q":
                break
            try:
                audio = record_take(sd, device)
            except sd.PortAudioError as e:
                print(f"Couldn't record: {e}\n{PERMISSION_HINT}")
                continue
            duration = len(audio) / MIC_SR
            if duration < MIN_TAKE_S:
                print("Too short, try again.")
                continue

            take_num += 1
            snr = squim(audio, MIC_SR)["snr"]
            print(f"Take {take_num}: {duration:.1f} s, est. SNR {snr:.1f} dB")
            for name, denoiser in systems.items():
                start = time.perf_counter()
                out = audio if denoiser is None else denoiser.denoise(audio, MIC_SR)
                text = transcribe(asr, resample(out, MIC_SR, WHISPER_SR), args.language)
                print(f"[{name} {time.perf_counter() - start:.1f}s] {text}")
                if not args.no_save:
                    sf.write(args.save_dir / f"take_{take_num:03d}_{name}.wav", np.clip(out, -1.0, 1.0), MIC_SR)
            if not args.no_save:
                print(f"   -> {args.save_dir}/take_{take_num:03d}_*.wav")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()

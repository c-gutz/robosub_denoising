"""Real-time FRCRN on the laptop mic.

Denoises the mic while you talk and plays the result to your headphones about
1.5 s behind (one block + lookahead + model time). Press Enter to switch what you
hear between denoised and raw (lined up, so it's a fair A/B); q + Enter quits.
On quit both versions are saved as .wav, and --transcribe prints Whisper's
transcript of each. --spectrogram opens a window with live spectrograms of the
raw and denoised audio.

Use headphones: through speakers the output feeds back into the mic.

Examples:
    python live_stream.py
    python live_stream.py --output-device "External Headphones"
    python live_stream.py --spectrogram               # also watch raw vs denoised live
    python live_stream.py --no-monitor --transcribe   # just record + denoise live
    python live_stream.py --file recordings/recording_4/audio_4.MOV --transcribe
                                                      # stream a file through, no mic
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

from denoising.streaming import StreamingFRCRN

SR = 16_000
CHUNK = 1600  # samples per audio callback (0.1 s)
PERMISSION_HINT = ("If the mic can't be opened, allow your terminal app under System Settings -> "
                   "Privacy & Security -> Microphone, or pick another --device (see --list-devices).")


class Fifo:
    """Thread-safe sample queue; reads past the end are padded with silence.

    The first write is preceded by ``cushion`` samples of silence, so playback
    runs that far behind and small hiccups in model time don't cause dropouts.
    """

    def __init__(self, prefill: int = 0, cushion: int = 0):
        self._chunks, self._size, self._lock = deque(), 0, threading.Lock()
        self.cushion, self.started = cushion, False
        self.underruns = 0  # reads that ran out after playback started
        self.total_read = 0  # samples read so far, including the cushion
        if prefill:
            self.write(np.zeros(prefill, dtype=np.float32))

    def __len__(self) -> int:
        return self._size

    def write(self, samples: np.ndarray) -> None:
        if len(samples):
            with self._lock:
                if not self.started and self.cushion:
                    self._chunks.append(np.zeros(self.cushion, dtype=np.float32))
                    self._size += self.cushion
                self.started = True
                self._chunks.append(samples)
                self._size += len(samples)

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            if not self.started:
                return out
            while filled < n and self._chunks:
                chunk = self._chunks.popleft()
                take = min(n - filled, len(chunk))
                out[filled: filled + take] = chunk[:take]
                if take < len(chunk):
                    self._chunks.appendleft(chunk[take:])
                filled += take
            self._size -= filled
            self.total_read += filled
            if filled < n:
                self.underruns += 1
        return out


def parse_device(value: str | None):
    return int(value) if value and value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", help="Input device index or name (default: system mic)")
    parser.add_argument("--output-device", help="Output device index or name (default: system output)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--no-monitor", action="store_true", help="Don't play anything back")
    parser.add_argument("--spectrogram", action="store_true",
                        help="Show live spectrograms of the raw and denoised audio (mic only)")
    parser.add_argument("--file", type=Path, help="Stream this file through instead of the mic (no audio devices)")
    parser.add_argument("--checkpoint", help="FRCRN checkpoint dir, e.g. fine-tuned weights (default: pretrained)")
    parser.add_argument("--block", type=float, default=0.8, help="Seconds of new audio per model call (default 0.8)")
    parser.add_argument("--context", type=float, default=0.4, help="Seconds of past audio the model sees (default 0.4)")
    parser.add_argument("--lookahead", type=float, default=0.16, help="Seconds of future audio the model sees (default 0.16)")
    parser.add_argument("--transcribe", action="store_true", help="On quit, print Whisper transcripts of raw and denoised")
    parser.add_argument("--whisper", default="openai/whisper-small", help="Hugging Face Whisper checkpoint")
    parser.add_argument("--language", default="english", help="Spoken language (ignored by .en checkpoints)")
    parser.add_argument("--save-dir", type=Path, default=Path("takes"), help="Where to save the .wav files")
    parser.add_argument("--no-save", action="store_true", help="Don't save anything")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    from denoising.models.frcrn import FRCRNDenoiser

    print("Loading FRCRN...")
    streamer = StreamingFRCRN(FRCRNDenoiser(checkpoint_dir=args.checkpoint),
                              block_s=args.block, context_s=args.context, lookahead_s=args.lookahead)
    streamer.process(np.zeros(streamer.block, dtype=np.float32))  # warm up
    streamer.reset()
    asr = None
    if args.transcribe:
        import transformers
        from transcribe import load_whisper
        transformers.logging.set_verbosity_error()
        print(f"Loading {args.whisper}...")
        asr = load_whisper(args.whisper)

    if args.file:
        if args.spectrogram:
            print("--spectrogram only works with the mic; ignoring it for --file.")
        raw, denoised = stream_file(streamer, args.file)
    else:
        raw, denoised = stream_mic(streamer, args)
    if len(raw) < SR // 2:
        print("Nothing recorded.")
        return

    if not args.no_save:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        n = len(list(args.save_dir.glob("stream_*_raw.wav"))) + 1
        for name, audio in [("raw", raw), ("frcrn_se_16k", denoised)]:
            sf.write(args.save_dir / f"stream_{n:03d}_{name}.wav", np.clip(audio, -1.0, 1.0), SR)
        print(f"Saved {args.save_dir}/stream_{n:03d}_{{raw,frcrn_se_16k}}.wav ({len(raw) / SR:.1f} s)")

    if asr is not None:
        from transcribe import transcribe
        for name, audio in [("raw", raw), ("frcrn_se_16k", denoised)]:
            print(f"[{name}] {transcribe(asr, audio, args.language)}")


def stream_file(streamer: StreamingFRCRN, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Feed a file through the streamer in mic-sized chunks, as fast as it will go."""
    from denoising.audio_io import load_audio
    from denoising.base import resample, to_mono_float

    audio, sr = load_audio(path)
    raw = resample(to_mono_float(audio), sr, SR)
    start = time.perf_counter()
    out = [streamer.process(raw[i: i + CHUNK]) for i in range(0, len(raw), CHUNK)]
    out.append(streamer.flush())
    took = time.perf_counter() - start
    print(f"Streamed {len(raw) / SR:.1f} s in {took:.1f} s ({took * SR / len(raw):.2f}x real time; "
          f"live needs < 1). Delay: {streamer.delay_s:.2f} s + model time.")
    denoised = np.concatenate(out)[streamer.lookahead:]  # undo the lookahead delay so it lines up with raw
    return raw, denoised


def stream_mic(streamer: StreamingFRCRN, args) -> tuple[np.ndarray, np.ndarray]:
    import sounddevice as sd

    mic = queue.Queue()
    played = Fifo(cushion=int(0.2 * SR))
    # Raw audio delayed by the same lookahead as the denoised output, for the A/B switch.
    raw_delay = Fifo(prefill=streamer.lookahead)
    raw_takes, denoised_takes = [], []
    aligned_raw_takes = []  # raw, lined up sample for sample with denoised_takes
    state = {"denoised": True, "busy": 0.0, "audio": 0.0, "captured": 0}
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set() or not mic.empty():
            try:
                chunk = mic.get(timeout=0.1)
            except queue.Empty:
                continue
            raw_takes.append(chunk)
            raw_delay.write(chunk)
            start = time.perf_counter()
            out = streamer.process(chunk)
            if len(out):
                state["busy"] += time.perf_counter() - start
                state["audio"] += len(out) / SR
                denoised_takes.append(out)
                aligned_raw = raw_delay.read(len(out))
                aligned_raw_takes.append(aligned_raw)
                if not args.no_monitor:
                    played.write(out if state["denoised"] else aligned_raw)

    def callback(indata, outdata, frames, time_info, status) -> None:
        mic.put(indata[:, 0].copy())
        state["captured"] += frames
        if outdata is not None:
            outdata[:] = played.read(frames)[:, None]

    def status_line() -> None:
        while not stop.wait(1.0):
            rtf = state["busy"] / state["audio"] if state["audio"] else 0.0
            behind = mic.qsize() * CHUNK / SR
            if args.no_monitor:
                heard = ""
            elif played.started:
                # Output sample i is input sample i - lookahead, so this is how far
                # what you hear now trails what the mic is picking up now.
                output_read = played.total_read - played.cushion
                delay = (state["captured"] - output_read + streamer.lookahead) / SR
                heard = (f"hearing {'DENOISED' if state['denoised'] else 'raw'} ~{delay:.1f} s behind | "
                         f"dropouts {played.underruns} | ")
            else:
                heard = "starting... | "
            warn = "  FALLING BEHIND: raise --block or lower --context" if behind > streamer.block / SR else ""
            sys.stderr.write(f"\r{heard}model {rtf:.2f}x real time{warn}   ")
            sys.stderr.flush()

    device = parse_device(args.device)
    try:
        if args.no_monitor:
            stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=CHUNK, device=device,
                                    callback=lambda i, f, t, s: callback(i, None, f, t, s))
        else:
            # Play the mono signal on both ears: a 1-channel stream can come out left-only.
            output = parse_device(args.output_device)
            out_channels = min(2, sd.query_devices(output, "output")["max_output_channels"])
            stream = sd.Stream(samplerate=SR, channels=(1, out_channels), dtype="float32", blocksize=CHUNK,
                               device=(device, output), callback=callback)
    except (sd.PortAudioError, ValueError) as e:
        print(f"Couldn't open audio: {e}\n{PERMISSION_HINT}")
        return np.zeros(0), np.zeros(0)

    threads = [threading.Thread(target=f, daemon=True) for f in (worker, status_line)]
    for t in threads:
        t.start()
    print(f"Listening. Block {streamer.block / SR:.2f} s, context {streamer.context / SR:.2f} s, "
          f"lookahead {streamer.lookahead / SR:.2f} s.")
    print("Enter: switch denoised/raw   q + Enter: quit" if not args.no_monitor else "q + Enter (or Ctrl+C): quit")
    quit_event = threading.Event()

    def toggle() -> None:
        state["denoised"] = not state["denoised"]

    def read_terminal() -> None:
        # sys.stdin.readline, not input(): with the spectrogram open, input() runs
        # matplotlib's macOS GUI hook on this thread, and macOS kills the process
        # (SIGTRAP) for touching the GUI off the main thread.
        try:
            while not quit_event.is_set():
                line = sys.stdin.readline()
                if not line or line.strip().lower() == "q":  # "" means stdin closed
                    break
                toggle()
        except OSError:
            pass
        quit_event.set()

    # Not joined: it may still be waiting for a line when the window is closed.
    threading.Thread(target=read_terminal, daemon=True).start()
    with stream:
        try:
            if args.spectrogram:
                # GUI windows have to run on the main thread on macOS.
                view = LiveSpectrogram(monitor=not args.no_monitor)
                view.run(lambda: (aligned_raw_takes, denoised_takes), quit_event, toggle,
                         lambda: state["denoised"])
            else:
                while not quit_event.wait(0.2):
                    pass
        except KeyboardInterrupt:
            pass
    quit_event.set()
    stop.set()
    for t in threads:
        t.join()
    sys.stderr.write("\n")

    raw = np.concatenate(raw_takes) if raw_takes else np.zeros(0, dtype=np.float32)
    denoised = np.concatenate(denoised_takes + [streamer.flush()])[streamer.lookahead:][: len(raw)]
    return raw, denoised


class LiveSpectrogram:
    """Matplotlib window with scrolling spectrograms of raw (top) and denoised (bottom).

    Both panels show the same moment, lined up sample for sample, on one fixed dB
    scale, so a darker background on the bottom means noise was removed.
    """

    N_FFT, HOP = 512, 160  # 32 ms frames every 10 ms
    SECONDS = 6.0
    DB_RANGE = (-100, -20)  # dBFS per bin, the same on both panels

    def __init__(self, monitor: bool):
        import matplotlib.pyplot as plt

        self.plt, self.monitor = plt, monitor
        self.window = np.hanning(self.N_FFT).astype(np.float32)
        self.n_samples = int(self.SECONDS * SR)
        n_frames = 1 + (self.n_samples - self.N_FFT) // self.HOP
        blank = np.full((self.N_FFT // 2 + 1, n_frames), self.DB_RANGE[0], dtype=np.float32)

        self.fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 6.5),
                                      constrained_layout=True)
        self.images = []
        for ax, title in zip(axes, ["Raw (before)", "FRCRN (after)"]):
            image = ax.imshow(blank, origin="lower", aspect="auto", cmap="magma",
                              vmin=self.DB_RANGE[0], vmax=self.DB_RANGE[1],
                              extent=(-self.SECONDS, 0, 0, SR / 2000))
            ax.set_title(title, loc="left")
            ax.set_ylabel("kHz")
            self.images.append(image)
        axes[-1].set_xlabel("seconds (newest at right; both lag the mic by the denoising delay)")
        self.fig.colorbar(self.images[0], ax=axes, label="dBFS", shrink=0.9)

    def run(self, get_takes, quit_event: threading.Event, toggle, is_denoised, fps: float = 10) -> None:
        """Redraw until the window is closed, q is pressed, or quit_event is set."""
        def on_key(event) -> None:
            if event.key in ("enter", " ") and self.monitor:
                toggle()
            # q closes the window (matplotlib's default), which sets quit_event below.

        self.fig.canvas.mpl_connect("key_press_event", on_key)
        self.fig.canvas.mpl_connect("close_event", lambda _: quit_event.set())
        self.plt.show(block=False)
        shown = None
        while not quit_event.is_set():
            raw_takes, denoised_takes = get_takes()
            n = min(len(raw_takes), len(denoised_takes))  # the worker may be mid-append
            # New audio only arrives once per block, so only redraw then (or on a switch):
            # drawing competes with the model for CPU.
            if (n, is_denoised()) != shown:
                shown = (n, is_denoised())
                for image, takes in zip(self.images, (raw_takes[:n], denoised_takes[:n])):
                    image.set_data(self._spectrogram(self._tail(takes)))
                heard = (f"   (hearing {'denoised' if is_denoised() else 'raw'}; Enter/space switches)"
                         if self.monitor else "")
                self.fig.suptitle(f"Live spectrogram{heard}   q or close window to quit", fontsize=10)
                self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(1 / fps)
        self.plt.close(self.fig)

    def _tail(self, takes: list[np.ndarray]) -> np.ndarray:
        """The last SECONDS of a list of chunks, left-padded with silence."""
        chunks, total = [], 0
        for chunk in reversed(takes):
            chunks.append(chunk)
            total += len(chunk)
            if total >= self.n_samples:
                break
        audio = np.concatenate(chunks[::-1]) if chunks else np.zeros(0, dtype=np.float32)
        audio = audio[-self.n_samples:]
        return np.pad(audio, (self.n_samples - len(audio), 0))

    def _spectrogram(self, audio: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(audio, self.N_FFT)[:: self.HOP]
        magnitude = np.abs(np.fft.rfft(frames * self.window, axis=1)) * (2 / self.window.sum())
        return (20 * np.log10(magnitude + 1e-10)).T  # (freq, time)


if __name__ == "__main__":
    main()

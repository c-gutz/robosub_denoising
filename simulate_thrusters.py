"""Simulate thruster noise heard by a microphone above the water, and mix it into recordings.

There are no recordings of our thrusters yet, so this synthesizes them from how a
brushless thruster like the Blue Robotics T200 (3 blades, 7 motor pole pairs) makes sound:

    blade tones    shaft rate x 3 blades, plus harmonics
    motor whine    harmonics of the electrical frequency (shaft rate x 7 pole pairs),
                   strongest near 6x and 12x, the commutation rate
    ESC whine      sometimes a fixed PWM tone at 8 or 16 kHz (only heard above 16 kHz sample rates)
    prop wash      broadband churn that pulses at the blade rate
    bubbles        bubble pops and splashes (submerged thrusters only)

Throttle holds and ramps like the sub maneuvering, with the propeller's inertia smoothing
each change. Thrusters in the same group (e.g. the vertical ones) share a throttle but spin
at slightly different speeds, so their tones beat. The sum goes through a random EQ (hull
resonances, distance, the phone mic's low cut) and reverb (pool hall). Every clip draws new
random settings, so many clips give varied noise.

Conditions:
    submerged  thrusters in the pool, mic in the air above (default): motor whine and churn,
               highs muffled by the water surface, bubbles
    air        thrusters spinning out of the water on a bench: loud blade tones and prop wash

This is a stand-in until there are real recordings. Listen to the output, and once you can,
record the thrusters running with nobody talking and compare.

Examples:
    python simulate_thrusters.py recordings/recording_4/audio_4.MOV            # mix at 0 dB SNR
    python simulate_thrusters.py recordings/*/*.MOV --snr 5 0 -5 --video       # also write .MOVs
    python simulate_thrusters.py --noise-only 20 --duration 30 -o thruster_noise
    python snr_sweep.py quiet_room.wav --noise thruster_noise/*.wav
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from denoising.audio_io import load_audio, mix_at_snr
from denoising.base import to_mono_float

BLADES = 3
POLE_PAIRS = 7
MAX_RPM = 3800       # T200 top speed in water is roughly 3500 RPM at 16 V
CONTROL_SR = 200     # throttle and speed curves are smooth: build them at a low rate, then interpolate
CONDITIONS = ("submerged", "air")


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)) + 1e-12)


def smooth_noise(n: int, rng: np.random.Generator, cutoff_hz: float) -> np.ndarray:
    """Unit-variance random wobble at the control rate, slower than ``cutoff_hz``."""
    b, a = signal.butter(1, cutoff_hz, fs=CONTROL_SR)
    x = signal.filtfilt(b, a, rng.standard_normal(n))
    return x / (x.std() + 1e-12)


def throttle_curve(n: int, rng: np.random.Generator) -> np.ndarray:
    """One thruster group's throttle (0-1) at the control rate: holds, stops and ramps."""
    targets = np.empty(n)
    i = 0
    while i < n:
        hold = int(rng.uniform(0.5, 4.0) * CONTROL_SR)
        targets[i:i + hold] = 0.0 if rng.random() < 0.15 else rng.uniform(0.15, 1.0)
        i += hold
    # The propeller can't change speed instantly; a first-order lag smooths each step.
    a = np.exp(-1 / (rng.uniform(0.08, 0.4) * CONTROL_SR))
    b, a = [1 - a], [1, -a]
    return signal.lfilter(b, a, targets, zi=signal.lfilter_zi(b, a) * targets[0])[0]


def thruster(rpm: np.ndarray, sr: int, rng: np.random.Generator, condition: str) -> np.ndarray:
    """One thruster's sound, given its shaft speed in RPM at every audio sample."""
    f_shaft = rpm / 60
    phase = 2 * np.pi * np.cumsum(f_shaft) / sr + rng.uniform(0, 2 * np.pi)
    limit = 0.45 * sr

    def harmonics(multiple: int, count: int, gain) -> np.ndarray:
        out = np.zeros_like(rpm)
        for k in range(1, count + 1):
            freq = multiple * k * f_shaft
            if freq.min() >= limit:
                break
            # Fade each harmonic out as it nears Nyquist so speed-ups don't alias.
            fade = np.clip((limit - freq) / (0.1 * limit), 0, 1)
            amp = gain(k) * rng.uniform(0.5, 1.5)
            out += amp * fade * np.sin(multiple * k * phase + rng.uniform(0, 2 * np.pi))
        return out / rms(out)

    blade = harmonics(BLADES, 30, lambda k: k**-1.0) + 0.15 * harmonics(1, 4, lambda k: k**-1.0)
    whine = harmonics(POLE_PAIRS, 24, lambda k: 0.3 * k**-0.5 + np.exp(-((k - 6) / 1.5) ** 2)
                      + 0.5 * np.exp(-((k - 12) / 2) ** 2))

    # Prop wash: band-limited noise that pulses each time a blade passes.
    top = min(3000 if condition == "submerged" else 8000, limit)
    sos = signal.butter(2, [100, top], btype="bandpass", fs=sr, output="sos")
    wash = signal.sosfilt(sos, rng.standard_normal(len(rpm)))
    wash *= 1 + rng.uniform(0.2, 0.7) * np.cos(BLADES * phase)
    wash /= rms(wash)

    if condition == "submerged":   # the water carries the motor; the blades mostly churn
        levels = (rng.uniform(0.1, 0.5), rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5))
    else:                          # spinning in air, the blades are the loudest part
        levels = (rng.uniform(1.0, 2.0), rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0))
    sound = levels[0] * blade + levels[1] * whine + levels[2] * wash

    if rng.random() < 0.5:
        pwm = rng.choice([8000, 16000])
        if pwm < limit:
            sound += rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * pwm * np.arange(len(rpm)) / sr)

    # Louder the faster it spins (acoustic power rises steeply with speed).
    return sound * (rpm / MAX_RPM) ** 2


def bubbles(load: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Bubble pops and splashes from churned water; more of them at high throttle."""
    n = len(load)
    out = np.zeros(n)
    pop_rate = rng.uniform(5, 40) * load**2           # events per second
    splash_rate = rng.uniform(0.3, 3) * load**2
    hp = signal.butter(2, min(1000, 0.4 * sr), btype="highpass", fs=sr, output="sos")
    for i in np.flatnonzero(rng.random(n) < pop_rate / sr):
        # A bubble rings at its resonant pitch, which rises as it shrinks.
        f0, decay = log_uniform(rng, 300, min(4000, 0.4 * sr)), rng.uniform(0.003, 0.03)
        t = np.arange(int(5 * decay * sr)) / sr
        chirp = rng.uniform(0, 2) / t[-1]
        pop = np.exp(-t / decay) * np.sin(2 * np.pi * f0 * (t + chirp * t**2 / 2))
        out[i:i + len(pop)] += rng.uniform(0.2, 1.0) * pop[:n - i]
    for i in np.flatnonzero(rng.random(n) < splash_rate / sr):
        decay = rng.uniform(0.02, 0.15)
        t = np.arange(int(4 * decay * sr)) / sr
        splash = np.exp(-t / decay) * signal.sosfilt(hp, rng.standard_normal(len(t)))
        out[i:i + len(splash)] += rng.uniform(0.5, 2.0) * splash[:n - i]
    return out / rms(out) if out.any() else out


def peaking_eq(f0: float, gain_db: float, q: float, sr: int) -> tuple[list, list]:
    """Biquad peaking filter (RBJ Audio EQ Cookbook)."""
    a_gain, w0 = 10 ** (gain_db / 40), 2 * np.pi * f0 / sr
    alpha, cos_w0 = np.sin(w0) / (2 * q), np.cos(w0)
    return ([1 + alpha * a_gain, -2 * cos_w0, 1 - alpha * a_gain],
            [1 + alpha / a_gain, -2 * cos_w0, 1 - alpha / a_gain])


def propagate(x: np.ndarray, sr: int, rng: np.random.Generator, condition: str) -> np.ndarray:
    """Colour the sound on its way to the mic: muffling, hull resonances, mic low cut, reverb."""
    cutoff = rng.uniform(1500, 5000) if condition == "submerged" else rng.uniform(5000, 12000)
    sos = [signal.butter(2, rng.uniform(60, 150), btype="highpass", fs=sr, output="sos")]
    if cutoff < 0.45 * sr:
        sos.append(signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos"))
    x = signal.sosfilt(np.vstack(sos), x)
    for _ in range(rng.integers(1, 4)):
        b, a = peaking_eq(log_uniform(rng, 150, min(4000, 0.4 * sr)), rng.uniform(-6, 9),
                          rng.uniform(1, 6), sr)
        x = signal.lfilter(b, a, x)
    # Reverb: exponentially decaying noise, 60 dB down after rt60 seconds.
    rt60, wet = rng.uniform(0.2, 1.8), rng.uniform(0.05, 0.5)
    t = np.arange(int(rt60 * sr)) / sr
    ir = rng.standard_normal(len(t)) * 10 ** (-3 * t / rt60)
    ir /= np.sqrt(np.sum(ir**2))
    return (1 - wet) * x + wet * signal.fftconvolve(x, ir)[: len(x)]


def synthesize(duration: float, sr: int, rng: np.random.Generator, condition: str = "submerged",
               n_thrusters: int | None = None) -> tuple[np.ndarray, dict]:
    """Thruster noise, ``duration`` seconds at ``sr``, with freshly drawn random settings.

    Returns the float32 noise (RMS 0.1, or less if that would clip) and a summary of the settings drawn.
    """
    n = int(duration * sr)
    n_control = int(duration * CONTROL_SR) + 2
    t, t_control = np.arange(n) / sr, np.arange(n_control) / CONTROL_SR
    n_thrusters = n_thrusters or int(rng.integers(1, 9))
    n_groups = int(rng.integers(1, min(3, n_thrusters) + 1))
    throttles = [throttle_curve(n_control, rng) for _ in range(n_groups)]

    total, load = np.zeros(n), np.zeros(n)
    for i in range(n_thrusters):
        # Same group, slightly different speed: a few percent offset plus a slow wobble.
        speed = throttles[i % n_groups] * (1 + rng.normal(0, 0.03))
        speed *= 1 + rng.uniform(0.005, 0.02) * smooth_noise(n_control, rng, 1.0)
        rpm = np.interp(t, t_control, np.clip(speed, 0, 1.1) * MAX_RPM)
        total += rng.uniform(0.3, 1.0) * thruster(rpm, sr, rng, condition)   # nearer or farther
        load += rpm / MAX_RPM / n_thrusters
    if condition == "submerged":
        total += rng.uniform(0.2, 1.0) * rms(total) * bubbles(load, sr, rng)

    out = propagate(total, sr, rng, condition)
    out *= 0.1 / rms(out)
    out *= min(1.0, 0.95 / np.max(np.abs(out)))   # splashes can be spiky; don't clip the .wav
    settings = {"condition": condition, "thrusters": n_thrusters, "groups": n_groups,
                "mean_throttle": round(float(load.mean()), 2)}
    return out.astype(np.float32), settings


def write_video(src: Path, audio: np.ndarray, sr: int, dst: Path) -> None:
    """Copy ``src``'s video stream into ``dst`` with ``audio`` as its sound track."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-i", tmp.name,
                        "-map", "0:v?", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                        "-shortest", str(dst)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", type=Path, help="Recordings to add thruster noise to")
    parser.add_argument("--snr", nargs="+", type=float, default=[0.0],
                        help="Speech-to-thruster SNRs in dB, one output per value (default: 0)")
    parser.add_argument("--condition", choices=CONDITIONS, default="submerged")
    parser.add_argument("--thrusters", type=int, help="Number of thrusters (default: random 1-8)")
    parser.add_argument("--noise-only", type=int, metavar="N", help="Write N noise-only clips instead")
    parser.add_argument("--duration", type=float, default=30.0, help="Noise-only clip length in seconds")
    parser.add_argument("--sr", type=int, default=16_000, help="Noise-only sample rate")
    parser.add_argument("--video", action="store_true", help="Also write a .MOV with the noisy audio")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("thruster_sim"))
    parser.add_argument("--seed", type=int, default=0, help="Random seed; change it for different noise")
    args = parser.parse_args()
    if not args.inputs and not args.noise_only:
        parser.error("give recordings to mix into, or --noise-only N")

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.noise_only or 0):
        noise, settings = synthesize(args.duration, args.sr, rng, args.condition, args.thrusters)
        path = args.out_dir / f"thrusters_{args.condition}_{i:03d}.wav"
        sf.write(path, noise, args.sr)
        print(f"{path}  {settings}")

    for src in args.inputs:
        audio, sr = load_audio(src)
        audio = to_mono_float(audio)
        # One noise draw per recording, so its SNR versions differ only in level.
        noise, settings = synthesize(len(audio) / sr, sr, rng, args.condition, args.thrusters)
        print(f"{src}  {settings}")
        for snr_db in args.snr:
            mixture = mix_at_snr(audio, noise, snr_db)
            stem = args.out_dir / f"{src.stem}_thrusters_{snr_db:+g}dB"
            sf.write(stem.with_suffix(".wav"), mixture, sr)
            print(f"  {stem.with_suffix('.wav')}")
            if args.video:
                write_video(src, mixture, sr, stem.with_suffix(".MOV"))
                print(f"  {stem.with_suffix('.MOV')}")


if __name__ == "__main__":
    main()

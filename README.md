# robosub_denoising

Speech recognition in noisy environments (dining hall, windy beach, crowds, etc.):
Whisper, optionally with a speech-enhancement model in front of it.

## Setup

```bash
conda env create -f environment.yml
conda activate myenv
```

## Transcribe

Accepts audio or video files, including iPhone recordings (`.MOV`, `.m4a`).

```bash
python transcribe.py IMG_0123.MOV              # Whisper on the raw audio
python transcribe.py IMG_0123.MOV --denoise    # denoise first, then Whisper
python transcribe.py IMG_0123.MOV --compare    # print both transcripts
python transcribe.py IMG_0123.MOV --compare --dnsmos   # plus DNSMOS quality scores
python transcribe.py IMG_0123.MOV --whisper openai/whisper-large-v3-turbo
```

Denoising does not always help: enhancement artifacts can make Whisper worse, so use
`--compare` (and later the benchmark) to check on real recordings.

### DNSMOS

`--dnsmos` adds Microsoft's DNSMOS scores (1-5, higher is better, no clean reference needed):
`SIG` speech quality, `BAK` background noise (5 = none), `OVRL` overall (P.835), `P808` overall (P.808).
`--personalized` uses the personalized model, which also counts background talkers as noise
(good for babble like a dining hall). DNSMOS measures how the audio *sounds*, not whether Whisper
understands it, so a denoiser can raise DNSMOS while making transcripts worse; judge by both.

## Denoise only

```bash
python denoise_file.py IMG_0123.MOV                  # writes denoised/IMG_0123_deepfilternet3.wav
python denoise_file.py clips/*.wav --out-sr 16000
python denoise_file.py IMG_0123.MOV --dnsmos         # DNSMOS before/after
```

From Python:

```python
from denoising import load_denoiser

denoiser = load_denoiser("deepfilternet3")
clean = denoiser.denoise(audio, sr=48000, out_sr=16000)  # any input sr; resampled internally
```

## Benchmark

Compares the raw recording ("before") against each denoiser ("after") on the same clips,
scoring DNSMOS, Whisper word error rate (WER) and speed.

```bash
python benchmark.py clips/*.MOV                                  # raw vs every denoiser
python benchmark.py clips/*.MOV -m deepfilternet3 frcrn_se_16k   # only these
python benchmark.py clips/*.MOV --csv results.csv                # per-clip CSV with transcripts
```

To get WER, put a hand-typed transcript of what was actually said next to each clip,
with the same name and a `.txt` extension:

```
clips/
  dining_hall.MOV
  dining_hall.txt    # "hey robot, go to the red gate ..."
  beach.MOV          # no transcript: DNSMOS and RTF only
```

Casing and punctuation don't matter; transcripts and Whisper output both go through Whisper's
English text normalizer before scoring.

Output columns, one row per clip × system (`raw` = no denoising):

| Column | Meaning | Better |
|---|---|---|
| `SIG` / `BAK` / `OVRL` | DNSMOS speech quality / background noise / overall (1-5) | higher |
| `WER` | Whisper word error rate vs the `.txt` transcript (`-` if none) | lower |
| `RTF` | denoising time ÷ clip length (`< 1` is faster than real time) | lower |

The summary at the end averages DNSMOS and RTF over all clips and pools WER across the clips
that have transcripts. WER is the number that matters for the robot; DNSMOS only tells you how
the audio sounds.

Tips for recordings:
- Record the same kind of commands the robot will hear, in each environment you care about.
- Several short clips (10-60 s) give a steadier picture than one long one; a WER difference on
  a handful of words is mostly noise.
- RTF on the first clip includes model warm-up, so it reads high.

## Denoising models

| Name | Native SR | Status |
|---|---|---|
| `deepfilternet3` | 48 kHz | done |
| `frcrn_se_16k` | 16 kHz | done (ClearerVoice-Studio; weights cached in `~/.cache/clearvoice`) |
| `mossformer2_se_48k` | 48 kHz | done (ClearerVoice-Studio) |
| MossFormerGAN_SE_16K, SepFormer DNS4 | — | todo |

Pick a denoiser with `-m`, e.g. `python transcribe.py IMG_0123.MOV --compare -m frcrn_se_16k`.

## Layout

```
transcribe.py      Whisper transcription, optionally denoised (--compare for before/after)
denoise_file.py    write denoised .wav files
benchmark.py       DNSMOS + WER + speed across denoisers
denoising/         Denoiser interface, model backends, audio loading
evaluation/        DNSMOS and WER scoring used by the benchmark
```

To add a model, subclass `denoising.base.Denoiser` (set `name`, `native_sr`, implement `_enhance`) and register it in `denoising/__init__.py`.

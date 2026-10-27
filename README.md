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

## Denoising models

| Name | Native SR | Status |
|---|---|---|
| `deepfilternet3` | 48 kHz | done |
| `frcrn_se_16k` | 16 kHz | done (ClearerVoice-Studio; weights cached in `~/.cache/clearvoice`) |
| `mossformer2_se_48k` | 48 kHz | done (ClearerVoice-Studio) |
| MossFormerGAN_SE_16K, SepFormer DNS4 | — | todo |

Pick a denoiser with `-m`, e.g. `python transcribe.py IMG_0123.MOV --compare -m frcrn_se_16k`.

To add a model, subclass `denoising.base.Denoiser` (set `name`, `native_sr`, implement `_enhance`) and register it in `denoising/__init__.py`.

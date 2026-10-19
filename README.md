# Antalia-2 Mini Turkish TTS for ComfyUI

**Turkish text to speech** inside ComfyUI with **Antalia-2 Mini** ([cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini),
Apache-2.0): a small model (7.6M parameters, 46 MB for both weight files) with one synthetic male voice, 48 kHz output, and
synthesis far faster than real time, on the CPU as well. It reads raw Turkish text: numbers, dates, times, amounts, initialisms,
web addresses and English or brand words are turned into their spoken form by the package's own normaliser. The nodes call the
official `antalia-mini` package and never download anything.

> **Unofficial.** This node pack is an independent integration. It is not affiliated with or endorsed by PatientDesk AI (the
> authors of Antalia-2 Mini) or by Comfy Org.
>
> 日本語の概要は [README.ja.md](README.ja.md) にあります。

## What it does

| Node | Purpose |
| --- | --- |
| **Antalia-2 Mini Loader** | Loads the model from `models/antalia/<name>/` (fp32 or fp16 weights) onto ComfyUI's GPU or the CPU |
| **Antalia-2 Mini Speak (Turkish TTS)** | Text → `audio` (mono AUDIO), `spoken_text` (the text as the model reads it), `report` |

- `seed`: the same text, seed and settings give the same audio (on the same device).
- `speed` 0.95, `steps` 8, `cfg` 2.0 are the release defaults; 4 steps are faster and measurably less accurate.
- `sample_rate`: 48000 (native), or 24000 / 16000 / 8000 (telephone), resampled by the package.
- `eq`: the release's output EQ and de-esser (on by default); off gives the raw vocoder output.
- Long texts are split into sentence-sized chunks by the package and joined with short pauses.
- `spoken_text` shows the normaliser's reading, e.g. `Saat 15:30'da` → `saat on beş otuzda`.

Connect **Save Audio (Advanced)** or **Preview Audio** after it, or use the audio in a video workflow.

| Measured here (default settings, model loaded) | 3.4 s sentence | 17.7 s paragraph |
| --- | --- | --- |
| CPU (24 threads) | 0.06 s | 0.26 s |
| RTX 5090 | 0.06 s | 0.14 s |

The authors' evaluation (model card): intelligibility level with the other small Turkish model EMA Lightning (WER 1.02 vs 1.00 %
on Freya-TR-Eval), better in the 8 kHz telephone band, slightly lower full-band UTMOS.

## How far it was checked

Windows 11, ComfyUI v0.42.1, CPU and RTX 5090; details and raw numbers in [docs/VERIFICATION.md](docs/VERIFICATION.md):

- **Bit-identical to the official package.** Seven cases (dates and times, a paragraph with an amount, a web address and English
  words, 4 and 1 steps, no guidance, 16 and 8 kHz, EQ off, fp16 weights): the pack's audio equals that of
  `Antalia(model=<folder>).say(...)` run in a separate environment, sample for sample, 7 of 7 on the CPU and 7 of 7 on CUDA. The
  official package itself gives identical samples run to run on both.
- **Real ComfyUI queue** (CPU and RTX 5090): the shipped workflows through Save Audio (FLAC) give the official audio to within
  half a 16-bit step (the FLAC rounding), at 48 kHz and 8 kHz, with fp16 weights and another seed; the spoken text is the
  official normaliser's.
- **Clean install** and **CI** (GitHub Actions; Ubuntu and Windows; ComfyUI v0.38.0 and v0.42.1): the real model on the CPU,
  compared with the official package, ComfyUI's validation of the workflows, and a real ComfyUI queue.

**Not checked:** speech quality itself (the pack reproduces the official output; see the model card's evaluation), Apple MPS,
Linux and macOS outside CI.

## Install

1. ComfyUI with the V3 node API (**v0.38.0** or later) and Python 3.10+. A GPU is optional.
2. Clone this repository into `ComfyUI/custom_nodes/` and install its requirement (`antalia-mini==1.0.1`, which brings `cmudict`
   and `soundfile`) into ComfyUI's Python:

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-Antalia custom_nodes/ComfyUI-Antalia
   python -m pip install -r custom_nodes/ComfyUI-Antalia/requirements.txt
   ```

3. Download the model (46 MB; `--variant fp32` or `fp16` for one of them). The script pins the revision and verifies every file's
   SHA-256:

   ```
   python custom_nodes/ComfyUI-Antalia/tools/download_model.py --models-dir models/antalia
   ```

   Result: `models/antalia/antalia-mini/` with `config.json`, `acoustic(.fp16).safetensors`, `vocoder(.fp16).safetensors` and
   the model card, LICENSE and NOTICE. Another location works through `extra_model_paths.yaml` with the key `antalia`.

4. Restart ComfyUI and open `workflows/gui/speak.json` (48 kHz FLAC and the spoken text) or `speak_phone.json` (8 kHz).

## Notes and limits

- One voice, Turkish. English words inside Turkish text are read the way a Turkish speaker says them; whole English sentences are
  not what the model is for.
- The voice is synthetic (the model card: trained on speech generated with VoxCPM2, no recordings of real speakers). Please tell
  listeners when they hear a synthetic voice, as the authors ask.
- CPU and GPU give slightly different audio (43-64 dB SNR between them in the seven cases); each repeats itself exactly.
- Empty text, or text without anything readable, is an error rather than empty audio.

## Licenses

- This node pack's code: **MIT** ([LICENSE](LICENSE)).
- `antalia-mini` (pip dependency) and the model (downloaded only by `tools/download_model.py`): **Apache-2.0**, PatientDesk AI
  ([NOTICE](NOTICE)).

## Development

```
python -m pip install pytest ruff
python tools/download_model.py --models-dir <dir>/antalia
ANTALIA_MODELS_DIR=<dir>/antalia ANTALIA_COMFYUI_DIR=<ComfyUI checkout> python -m pytest tests
python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <dir>/antalia [--gpu]   # real queue
```

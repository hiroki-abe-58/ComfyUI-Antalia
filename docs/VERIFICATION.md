# Verification

What was compared, how, and the raw results. Machine: Windows 11 Pro (26200), NVIDIA RTX 5090 (driver 610.88), 24 CPU threads,
Python 3.12.13, PyTorch 2.14.1+cu130.

| | |
| --- | --- |
| Official package | `antalia-mini` 1.0.1 from PyPI (wheel SHA-256 `0dd5cd76…5644`, identical to the wheel in the model repository) |
| Model | cloud0day3/antalia-mini revision `8a79528770af2ed5cb6d3b9ac1248d370429f272`; every file's SHA-256 equals `tools/model_files.sha256`, and the weight files equal the hashes in the release's own config.json |
| Environments | "official": Python 3.12, PyTorch 2.14.1+cu130, antalia-mini 1.0.1 and its dependencies (numpy 2.5.4); "ComfyUI": a fresh ComfyUI v0.42.1 with its requirements and antalia-mini 1.0.1 (numpy 2.5.2) |

## Cases

`Antalia(model=<release folder>, device=..., variant=...).say(text, **kwargs)`, the README usage with a local folder.

| Case | Text | Arguments |
| --- | --- | --- |
| appointment | Randevunuz 14 Ekim Salı günü saat 15:30'da. | seed 0 |
| order | Siparişiniz yola çıktı. Kargo takip numaranız 1234 5678. Toplam tutar 249,90 TL. Detaylar için www.ornek.com.tr adresini ziyaret edin ve WhatsApp hattımızı arayın. | seed 7 |
| fast4 | Merhaba, size nasıl yardımcı olabilirim? | seed 1, speed 1.1, steps 4, cfg 1.5 |
| rate16_noeq | (appointment) | seed 0, sample_rate 16000, eq False |
| one_step_unguided | Öğleden sonra ıslak zeminde dikkatli yürüyün. | seed 3, steps 1, cfg 1.0 |
| rate8 | Kartınızın son dört hanesi 4821. Onaylıyor musunuz? | seed 11, sample_rate 8000 |
| fp16 | (appointment) | seed 0, fp16 weights |

The official calls pass only these arguments (the release defaults for the rest); the pack's calls pass the node's explicit
defaults (speed 0.95, steps 8, cfg 2.0, 48000, eq on), which are the release's.

## 1. The official package repeats itself

Two runs of the official environment: 7 of 7 cases sample-identical on the CPU, 7 of 7 on CUDA. CPU and CUDA differ from each
other (SNR 43.1-64.2 dB; the 17.7 s paragraph is the lowest).

## 2. The pack equals the official package

The pack's `runtime.speak()` in the ComfyUI environment against the official runs above, every sample compared:

| Device | Result |
| --- | --- |
| CPU | **7 / 7 identical** |
| CUDA (RTX 5090) | **7 / 7 identical** |

`spoken_text` equals `Antalia.normalize(text)`, e.g. "randevunuz on dört ekim salı günü saat on beş otuzda.".

## 3. Real ComfyUI queue

`scripts/ci_queue_check.py` on ComfyUI v0.42.1: a fresh base directory with the pack in `custom_nodes`, the model through
`extra_model_paths.yaml`, the shipped API workflows posted to `/prompt`. The reference is the official API in the check's own
process on the same device.

| Run | Through the queue | CPU | RTX 5090 |
| --- | --- | --- | --- |
| speak.json (48 kHz) | Save Audio (Advanced, FLAC) vs official | 0.5 of a 16-bit step at most | 0.5 |
| speak_phone.json (8 kHz) | same | 0.5 | 0.5 |
| speak.json, fp16 weights | same | 0.5 | 0.5 |
| speak.json, seed 9 | same; and differs from seed 0 | 0.5 | 0.5 |
| speak.json again (after other jobs) | the same file as the first run | identical | identical |

Half a step is the rounding of ComfyUI's 16-bit FLAC. The spoken text shown by PreviewAny equals `Antalia.normalize(text)`.
`../antalia-mini` as the model and 44100 as the sample rate were rejected by ComfyUI's validation; empty text failed with "the
text is empty". The UI workflows were exported from the real frontend (no missing nodes; the graph converts back to the API
workflows exactly).

## 4. Speed (model loaded, official API, default settings)

| Case | Audio | CPU (24 threads) | RTX 5090 |
| --- | --- | --- | --- |
| appointment | 3.40 s | 0.061 s | 0.058 s |
| order | 17.72 s | 0.26 s | 0.14 s |
| fast4 (4 steps) | 2.32 s | 0.037 s | 0.032 s |

## 5. Clean install

A fresh ComfyUI v0.42.1 checkout and a fresh Python 3.12 environment (PyTorch 2.14.1+cu130 and ComfyUI's requirements), then
`git clone` of this repository into `custom_nodes`, `pip install -r requirements.txt` (adds antalia-mini 1.0.1 with cmudict,
soundfile and their small helpers), and `tools/download_model.py --models-dir models/antalia`: 8 of 8 files downloaded at the
pinned revision and verified. The GPU real-queue check of section 3 against that installation (`--base-dir` = the ComfyUI folder)
passed with the same results; the cloned working tree was unchanged afterwards.

The Registry package (`comfy node validate` and `comfy node pack`, comfy-cli 1.22.0, on the committed tree): validation passed;
the archive holds the 18 files of the node (no tests, scripts or weights) and none of the scanned patterns (environment, network
or process code, absolute paths, tokens).

## 6. Tests and CI

`tests/` (32 tests) on the CPU with ComfyUI v0.42.1 and the real model, every reference required: the pack against the official
API in the same process (six argument sets, fp16), seeds, the package's checksum check on changed weights, missing files, the
pinned package version, the package scan (no environment, network or process code in what ships; the official resolver returns
a folder without downloading), ComfyUI's validation of every API workflow, the UI workflows against the API ones, node execution
against the runtime, invalid inputs, and the download tool.

CI (GitHub Actions, CPU, the real model fetched by the download tool) runs the same tests and the real-queue check on four jobs:
ComfyUI v0.38.0 on Ubuntu and Windows (Python 3.12, PyTorch 2.14.1), ComfyUI v0.38.0 on Ubuntu with Python 3.10 and PyTorch
2.8.0, and ComfyUI v0.42.1 on Ubuntu; all passed with nothing skipped.

## Not checked

Speech quality itself (see the model card's evaluation), Apple MPS, Linux and macOS outside CI.

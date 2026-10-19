"""Antalia-2 Mini (Turkish text to speech) with the official `antalia-mini` package (PyPI, cloud0day3/antalia-mini).

The steps are those of the official API (README and `antalia_mini.api`):

    tts = Antalia(model=<release folder>, device=..., variant="fp32")       # config.json + acoustic / vocoder safetensors
    speech = tts.say(text, seed=0, speed=0.95, steps=8, cfg=2.0, sample_rate=48000, eq=True)

`speak()` makes exactly that call and turns `speech.audio` (mono float32) into a ComfyUI AUDIO. The model always comes from a
local folder: the package downloads only when it is given a Hugging Face repo id instead of a folder, which this module never does.
"""

from __future__ import annotations

import os
import threading
import time

PACKAGE_VERSION = "1.0.1"
VARIANTS = ("fp32", "fp16")
SAMPLE_RATES = ("48000", "24000", "16000", "8000")  # antalia_mini.SAMPLE_RATES: 48 kHz native, the others resampled
DEFAULT_SPEED, DEFAULT_STEPS, DEFAULT_CFG = 0.95, 8, 2.0  # the release's config.json "defaults"
MAX_SEED = 2**31 - 1  # the package draws its random seeds below 2**31
_LOCK = threading.Lock()
_CACHE: dict = {}


def required_files(variant: str) -> tuple[str, ...]:
    """config.json and the two weight files of a variant, as the package names them."""
    from antalia_mini.weights import FILES

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return ("config.json", *FILES[variant])


def check_package() -> None:
    import antalia_mini

    if antalia_mini.__version__ != PACKAGE_VERSION:
        raise RuntimeError(f"this node pack was verified with antalia-mini {PACKAGE_VERSION}; installed: {antalia_mini.__version__} "
                           f"(pip install antalia-mini=={PACKAGE_VERSION})")


def load(folder: str, device: str = "cpu", variant: str = "fp32"):
    """An `antalia_mini.Antalia` on the release files in `folder` (it checks their SHA-256 against config.json); kept for the
    session per folder, device and variant."""
    missing = [n for n in required_files(variant) if not os.path.isfile(os.path.join(folder, n))]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} not found in {folder} (see the README for the download command)")
    check_package()
    key = (os.path.realpath(folder), str(device), variant)
    with _LOCK:
        if key not in _CACHE:
            from antalia_mini import Antalia

            _CACHE[key] = Antalia(model=os.path.realpath(folder), device=str(device), variant=variant)
        return _CACHE[key]


def release() -> None:
    with _LOCK:
        _CACHE.clear()


def speak(tts, text: str, seed: int = 0, speed: float = DEFAULT_SPEED, steps: int = DEFAULT_STEPS, cfg: float = DEFAULT_CFG,
          sample_rate: str = "48000", eq: bool = True) -> tuple[dict, str, dict]:
    """(AUDIO [1, 1, samples], the text as the model reads it, report) for `tts.say(text, ...)`."""
    import torch

    if sample_rate not in SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}")
    if not 0 <= int(seed) <= MAX_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_SEED}")
    if not text or not text.strip():
        raise ValueError("the text is empty")
    started = time.perf_counter()
    speech = tts.say(text, seed=int(seed), speed=float(speed), steps=int(steps), cfg=float(cfg), sample_rate=int(sample_rate), eq=bool(eq))
    seconds = time.perf_counter() - started
    if speech.audio.size == 0:
        raise ValueError("nothing to say: the text has no letters the model can read")
    spoken = tts.normalize(text)
    audio = {"waveform": torch.from_numpy(speech.audio.copy()).view(1, 1, -1), "sample_rate": speech.sample_rate}
    report = {"seed": speech.seed, "sample_rate": speech.sample_rate, "audio_seconds": round(speech.duration, 3),
              "seconds": round(seconds, 3), "device": str(tts.device), "speed": float(speed), "steps": int(steps), "cfg": float(cfg),
              "eq": bool(eq)}
    return audio, spoken, report

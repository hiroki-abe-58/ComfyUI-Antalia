"""The pack's speak() against the official `Antalia(...).say(...)` on the real model (CPU), bit for bit."""

import numpy as np
import pytest
import torch

from antalia_comfy import runtime

CASES = [
    ("Randevunuz 14 Ekim Salı günü saat 15:30'da.", {"seed": 0}),
    ("Siparişiniz yola çıktı. Toplam tutar 249,90 TL. Detaylar için www.ornek.com.tr adresini ziyaret edin.", {"seed": 7}),
    ("Merhaba, size nasıl yardımcı olabilirim?", {"seed": 1, "speed": 1.1, "steps": 4, "cfg": 1.5}),
    ("Randevunuz 14 Ekim Salı günü saat 15:30'da.", {"seed": 0, "sample_rate": 16000, "eq": False}),
    ("Öğleden sonra ıslak zeminde dikkatli yürüyün.", {"seed": 3, "steps": 1, "cfg": 1.0}),
    ("Kartınızın son dört hanesi 4821. Onaylıyor musunuz?", {"seed": 11, "sample_rate": 8000}),
]


@pytest.fixture(scope="module")
def tts(release_dir):
    yield runtime.load(str(release_dir), "cpu", "fp32")
    runtime.release()


def _pack_args(kw):
    return {"seed": kw["seed"], "speed": kw.get("speed", runtime.DEFAULT_SPEED), "steps": kw.get("steps", runtime.DEFAULT_STEPS),
            "cfg": kw.get("cfg", runtime.DEFAULT_CFG), "sample_rate": str(kw.get("sample_rate", 48000)), "eq": kw.get("eq", True)}


@pytest.mark.parametrize("text,kw", CASES)
def test_speak_equals_official_say(tts, official, text, kw):
    """The official call with only the given arguments (the release defaults for the rest) and the pack's call with the node's
    explicit defaults give the same samples."""
    expected = official.say(text, **kw)
    audio, spoken, report = runtime.speak(tts, text, **_pack_args(kw))
    assert audio["sample_rate"] == expected.sample_rate == kw.get("sample_rate", 48000)
    assert audio["waveform"].dtype == torch.float32 and audio["waveform"].shape == (1, 1, len(expected.audio))
    assert np.array_equal(audio["waveform"][0, 0].numpy(), expected.audio)
    assert spoken == official.normalize(text) and report["seed"] == kw["seed"]


def test_fp16_variant_equals_official(release_dir):
    from antalia_mini import Antalia

    text = "Randevunuz 14 Ekim Salı günü saat 15:30'da."
    expected = Antalia(model=str(release_dir), device="cpu", variant="fp16").say(text, seed=0)
    audio, _, _ = runtime.speak(runtime.load(str(release_dir), "cpu", "fp16"), text, seed=0)
    assert np.array_equal(audio["waveform"][0, 0].numpy(), expected.audio)


def test_seed_changes_the_audio_and_repeats(tts):
    a, _, _ = runtime.speak(tts, "Merhaba.", seed=1)
    b, _, _ = runtime.speak(tts, "Merhaba.", seed=2)
    c, _, _ = runtime.speak(tts, "Merhaba.", seed=1)
    assert torch.equal(a["waveform"], c["waveform"]) and not torch.equal(a["waveform"], b["waveform"])


def test_normalised_text():
    from antalia_mini import Antalia

    assert Antalia.normalize("Saat 15:30'da 3 kişi.") == "saat on beş otuzda üç kişi."


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_text_is_an_error(tts, text):
    with pytest.raises(ValueError, match="empty"):
        runtime.speak(tts, text)


def test_no_letters_is_an_error(tts):
    with pytest.raises(ValueError, match="nothing to say"):
        runtime.speak(tts, "... !!! ???")


def test_invalid_arguments(tts):
    with pytest.raises(ValueError):
        runtime.speak(tts, "Merhaba.", sample_rate="44100")
    with pytest.raises(ValueError):
        runtime.speak(tts, "Merhaba.", seed=-1)
    with pytest.raises(ValueError):
        runtime.required_files("int8")


def test_missing_files_are_reported(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="acoustic.safetensors"):
        runtime.load(str(tmp_path), "cpu", "fp32")


def test_changed_weights_are_rejected(release_dir, tmp_path):
    """The package checks the files against the SHA-256 in config.json."""
    import shutil

    folder = tmp_path / "broken"
    shutil.copytree(release_dir, folder)
    data = bytearray((folder / "vocoder.safetensors").read_bytes())
    data[-1] ^= 1
    (folder / "vocoder.safetensors").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        runtime.load(str(folder), "cpu", "fp32")


def test_the_pinned_package_is_installed():
    import antalia_mini

    assert antalia_mini.__version__ == runtime.PACKAGE_VERSION
    runtime.check_package()


def test_the_cache_keeps_one_instance_per_setting(release_dir):
    a = runtime.load(str(release_dir), "cpu", "fp32")
    assert runtime.load(str(release_dir), "cpu", "fp32") is a
    assert runtime.load(str(release_dir), "cpu", "fp16") is not a

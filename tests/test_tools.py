"""tools/download_model.py: the manifest covers the pinned release files, and verification catches missing or changed files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import download_model  # noqa: E402


def test_manifest_has_the_release_files():
    files = download_model.expected_files()
    assert set(files) == {"config.json", "acoustic.safetensors", "vocoder.safetensors", "acoustic.fp16.safetensors",
                          "vocoder.fp16.safetensors", "LICENSE", "NOTICE", "README.md"}
    assert all(len(d) == 64 for d in files.values())


def test_manifest_matches_the_release_config(models_dir):
    """config.json lists the weight files' SHA-256 itself; the manifest agrees with it."""
    import json

    config = json.loads((models_dir / "antalia-mini" / "config.json").read_text(encoding="utf-8"))
    files = download_model.expected_files()
    assert {k: v["sha256"] for k, v in config["files"].items()} == {k: v for k, v in files.items() if k.endswith(".safetensors")}


def test_verify_only_reports_missing_and_changed_files(tmp_path, capsys):
    folder = tmp_path / "antalia-mini"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert download_model.main(["--models-dir", str(tmp_path), "--variant", "fp16", "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH antalia-mini/config.json" in out and "MISSING  antalia-mini/acoustic.fp16.safetensors" in out
    assert "antalia-mini/acoustic.safetensors" not in out  # --variant fp16 skips the fp32 files


def test_verify_only_accepts_the_real_files(models_dir):
    assert download_model.main(["--models-dir", str(models_dir), "--verify-only"]) == 0

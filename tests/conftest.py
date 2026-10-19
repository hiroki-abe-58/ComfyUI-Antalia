"""Test setup. References come from environment variables, never from this repository:

ANTALIA_MODELS_DIR    a models/antalia folder holding antalia-mini/ (tools/download_model.py fetches it; 46 MB)
ANTALIA_COMFYUI_DIR   ComfyUI checkout (v0.38.0 or later)
ANTALIA_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
The tests run the real model on the CPU and compare the pack with the official `antalia_mini.Antalia(...).say(...)` in the same
process. The official package is this pack's dependency (pinned in requirements.txt).
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("ANTALIA_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("ANTALIA_COMFYUI_DIR")
MODELS = _path("ANTALIA_MODELS_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (ANTALIA_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (ANTALIA_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def models_dir():
    """The models/antalia folder (holding antalia-mini/)."""
    return Path(_need(MODELS, "Antalia model files (ANTALIA_MODELS_DIR)"))


@pytest.fixture(scope="session")
def release_dir(models_dir):
    return models_dir / "antalia-mini"


@pytest.fixture(scope="session")
def official(release_dir):
    """The official API on the same release folder, on the CPU (a separate instance from the pack's cached one)."""
    from antalia_mini import Antalia

    return Antalia(model=str(release_dir), device="cpu")

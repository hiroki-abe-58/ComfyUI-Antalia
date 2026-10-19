"""The installed package must not contain code that reads environment variables, opens network connections or starts processes (the
categories of the Comfy Registry's automated scan; scripts/ and tests/ are not part of a Registry package), and the nodes must
never download: only tools/download_model.py, run by the user, talks to Hugging Face. The official package (a pinned dependency)
downloads only when given a repo id, and the pack always passes it a folder."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "antalia_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)


def test_only_the_download_tool_knows_about_the_hub():
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "antalia_comfy").rglob("*.py") if "__pycache__" not in p.parts)
    assert not re.search(r"^\s*(from|import)\s+huggingface_hub|hf_hub_download|snapshot_download", text, re.M)
    assert "Antalia(model=os.path.realpath(folder)" in (ROOT / "antalia_comfy" / "runtime.py").read_text(encoding="utf-8")
    assert "hf_hub_download" in (ROOT / "tools" / "download_model.py").read_text(encoding="utf-8")


def test_a_folder_is_never_mistaken_for_a_repo_id(tmp_path):
    """The package's resolver returns an existing folder as is (no download, no variable lookup)."""
    from antalia_mini.api import _resolve

    assert _resolve(str(tmp_path), None, "fp32") == tmp_path


def test_comfyignore_keeps_tests_and_maintainer_tools_out_of_the_registry_package():
    lines = {line.strip() for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")}
    assert {"tests/", "scripts/", ".github/"} <= lines


def test_pyproject_and_requirements_agree_and_pin_the_package():
    from antalia_comfy import runtime

    deps = re.search(r"^dependencies = \[(.*)\]$", (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    reqs = [line.strip() for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    assert re.findall(r'"([^"]+)"', deps) == reqs == [f"antalia-mini=={runtime.PACKAGE_VERSION}"]

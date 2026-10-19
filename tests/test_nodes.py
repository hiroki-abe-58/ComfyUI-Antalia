"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and node
execution compared with the runtime (which tests/test_runtime.py compares with the official package)."""

import asyncio
import importlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"AntaliaLoader", "AntaliaSpeak"}


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `antalia_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaSpeak"].__module__.rsplit(".", 1)[0] + "." + name)


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, release_dir, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_input_directory(str(base / "input"))
    (base / "input").mkdir()
    models = base / "models"
    shutil.copytree(release_dir, models / "antalia-mini")
    (models / "not-a-model").mkdir()
    folder_paths.add_model_folder_path("antalia", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    yield nodes
    pack_module(nodes, "runtime").release()


def _model(comfy_nodes, variant="fp32"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaLoader"].execute("antalia-mini", "cpu", variant).args[0]


def test_registered_and_models_listed(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)
    schema = comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaLoader"].GET_SCHEMA()
    assert next(i for i in schema.inputs if i.id == "model").options == ["antalia-mini"]  # folders without a release are not offered


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def test_gui_workflows_match_api_workflows():
    """Every API workflow has a UI workflow with the same node types and widget values (exported by scripts/export_gui_workflows.py)."""
    for api_file in (ROOT / "workflows" / "api").glob("*.json"):
        api = json.loads(api_file.read_text(encoding="utf-8"))
        gui = json.loads((ROOT / "workflows" / "gui" / api_file.name).read_text(encoding="utf-8"))
        assert sorted(n["class_type"] for n in api.values()) == sorted(n["type"] for n in gui["nodes"])
        for node_id, node in api.items():
            ui = next(n for n in gui["nodes"] if str(n["id"]) == node_id)
            widgets = [v for v in node["inputs"].values() if not isinstance(v, list)]
            assert [w for w in ui.get("widgets_values", []) if w in widgets] == widgets, (api_file.name, node_id)


def test_speak_node_equals_runtime(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    model = _model(comfy_nodes)
    text = "Randevunuz 14 Ekim Salı günü saat 15:30'da."
    audio, spoken, report = comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaSpeak"].execute(model, text, 5, 0.95, 8, 2.0, "24000", True).args
    expected, expected_text, _ = runtime.speak(runtime.load(model["folder"], "cpu", "fp32"), text, 5, 0.95, 8, 2.0, "24000", True)
    assert np.array_equal(audio["waveform"].numpy(), expected["waveform"].numpy()) and audio["sample_rate"] == 24000
    assert spoken == expected_text == "randevunuz on dört ekim salı günü saat on beş otuzda."
    assert json.loads(report)["seed"] == 5


def test_invalid_inputs_are_rejected(comfy_nodes):
    loader = comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaLoader"]
    with pytest.raises(FileNotFoundError):
        loader.execute("../antalia-mini", "cpu", "fp32")
    with pytest.raises(FileNotFoundError):
        loader.execute("not-a-model", "cpu", "fp32")
    with pytest.raises(ValueError):
        loader.execute("antalia-mini", "cuda:7", "fp32")
    with pytest.raises(ValueError):
        loader.execute("antalia-mini", "cpu", "int8")
    speak = comfy_nodes.NODE_CLASS_MAPPINGS["AntaliaSpeak"]
    with pytest.raises(ValueError):
        speak.execute(_model(comfy_nodes), "  ", 0, 0.95, 8, 2.0, "48000", True)
    with pytest.raises(ValueError):
        speak.execute(_model(comfy_nodes), "Merhaba.", 0, 0.95, 8, 2.0, "44100", True)

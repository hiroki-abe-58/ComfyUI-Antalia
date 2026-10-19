"""ComfyUI nodes for Antalia-2 Mini, a small Turkish text-to-speech model (nothing is downloaded at run time)."""

from __future__ import annotations

import json

from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

AntaliaModel = io.Custom("ANTALIA_MODEL")
DEVICES = ("auto", "cpu")


def _device(choice: str) -> str:
    import comfy.model_management as mm

    if choice == "cpu":
        return "cpu"
    device = mm.get_torch_device()
    return str(device) if device.type in ("cuda", "mps") else "cpu"


class AntaliaLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AntaliaLoader",
            display_name="Antalia-2 Mini Loader",
            category="loaders/antalia",
            description=("Loads Antalia-2 Mini (Turkish text to speech, 7.6M parameters, 48 kHz) from models/antalia/<name>/, the release "
                         "files of the Hugging Face repository cloud0day3/antalia-mini."),
            inputs=[
                io.Combo.Input("model", options=paths.model_names(), tooltip="a folder in models/antalia/"),
                io.Combo.Input("device", options=list(DEVICES), default="auto",
                               tooltip="auto: ComfyUI's GPU (CUDA or Apple MPS), otherwise the CPU. The model is small; the CPU is fast too"),
                io.Combo.Input("variant", options=list(runtime.VARIANTS), default="fp32",
                               tooltip="weight files: fp32, or fp16 (half the download; computed in fp32 either way, practically the same audio)"),
            ],
            outputs=[AntaliaModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, device, variant) -> io.NodeOutput:
        if device not in DEVICES or variant not in runtime.VARIANTS:
            raise ValueError(f"unknown device {device!r} or variant {variant!r}")
        folder = paths.model_dir(model)
        dev = _device(device)
        runtime.load(folder, dev, variant)  # fail here, not later, when a file is missing or its checksum is wrong
        return io.NodeOutput({"folder": folder, "device": dev, "variant": variant})


class AntaliaSpeak(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AntaliaSpeak",
            display_name="Antalia-2 Mini Speak (Turkish TTS)",
            category="audio/antalia",
            description=("Speaks Turkish text with Antalia-2 Mini's one synthetic male voice. Raw text is fine: numbers, dates, times, "
                         "amounts, initialisms, web addresses and English or brand words are normalised by the package. The same text, "
                         "seed and settings give the same audio."),
            inputs=[
                AntaliaModel.Input("model"),
                io.String.Input("text", multiline=True, default="Merhaba, size nasıl yardımcı olabilirim?"),
                io.Int.Input("seed", default=0, min=0, max=runtime.MAX_SEED, control_after_generate=True),
                io.Float.Input("speed", default=runtime.DEFAULT_SPEED, min=0.5, max=2.0, step=0.01,
                               tooltip="speaking rate; 0.95 is the release default (slightly slower than the voice's own pace)"),
                io.Int.Input("steps", default=runtime.DEFAULT_STEPS, min=1, max=32,
                             tooltip="flow-matching steps; 8 is the release default, 4 is faster and measurably less accurate"),
                io.Float.Input("cfg", default=runtime.DEFAULT_CFG, min=1.0, max=5.0, step=0.1, tooltip="classifier-free guidance; 1 turns it off"),
                io.Combo.Input("sample_rate", options=list(runtime.SAMPLE_RATES), default="48000",
                               tooltip="48000 is native; 24000 / 16000 / 8000 (telephone) are resampled by the package"),
                io.Boolean.Input("eq", default=True, tooltip="the release's output EQ and de-esser; off gives the raw vocoder output"),
            ],
            outputs=[io.Audio.Output(display_name="audio"), io.String.Output(display_name="spoken_text"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, text, seed, speed, steps, cfg, sample_rate, eq) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        tts = runtime.load(model["folder"], model["device"], model["variant"])
        audio, spoken, report = runtime.speak(tts, text, seed, speed, steps, cfg, sample_rate, eq)
        report = json.dumps({"variant": model["variant"], **report}, indent=1)
        return io.NodeOutput(audio, spoken, report)


class AntaliaExtension(ComfyExtension):
    async def get_node_list(self):
        return [AntaliaLoader, AntaliaSpeak]


async def comfy_entrypoint() -> AntaliaExtension:
    paths.register()
    return AntaliaExtension()

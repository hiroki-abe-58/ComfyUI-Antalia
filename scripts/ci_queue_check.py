"""Real-queue check for CI, for the GPU verification and for a clean-install test: start a real ComfyUI server with this node pack
installed, and run the shipped workflows through POST /prompt with the real model.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> --models-dir <folder holding antalia-mini/> [--gpu]
                                     [--workdir DIR] [--base-dir DIR]

The reference is the official API in this process, `antalia_mini.Antalia(model=<folder>, device=...).say(text, seed=...)`, with
each workflow's text and settings. Through the queue
- Antalia Speak -> Save Audio (FLAC) gives the official audio to within one 16-bit step (ComfyUI's FLAC is 16 bits), at 48 kHz and
  at the telephone rate of 8 kHz, and the spoken text (PreviewAny) is `Antalia.normalize(text)`;
- the fp16 weights give the official fp16 audio, another seed gives other audio, the same job again gives the same file;
- invalid inputs are rejected.
The server is started and stopped by this script only. --base-dir reuses a ComfyUI base directory that already has the pack in
custom_nodes (clean-install test); otherwise a fresh one is made in the work directory with a copy of this checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import queue_client  # noqa: E402

NODE_IDS = {"AntaliaLoader", "AntaliaSpeak"}
STEP = 1 / 32768  # one 16-bit step


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def main(argv=None) -> int:
    import soundfile as sf
    from antalia_mini import Antalia

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True, help="a folder holding antalia-mini/")
    parser.add_argument("--gpu", action="store_true", help="run ComfyUI and the reference on CUDA (default: --cpu)")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--base-dir", type=Path, default=None, help="a ComfyUI base directory with the pack installed (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    models = args.models_dir.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="antalia-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base = (args.base_dir or work / "base").resolve()
    pack = base / "custom_nodes" / "ComfyUI-Antalia"
    if args.base_dir is None:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--base-dir: {pack} is not installed")
    output = base / "output"
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"antalia_ci:\n  base_path: {models.parent.as_posix()}\n  antalia: {models.name}\n", encoding="utf-8")

    workflows = {n: json.loads((ROOT / "workflows" / "api" / f"{n}.json").read_text(encoding="utf-8")) for n in ("speak", "speak_phone")}
    device = "cuda" if args.gpu else "cpu"
    release = str(models / "antalia-mini")
    official = {v: Antalia(model=release, device=device, variant=v) for v in ("fp32", "fp16")}

    def reference(name: str, variant: str = "fp32", **overrides):
        s = dict(workflows[name]["2"]["inputs"], **overrides)
        return official[variant].say(s["text"], seed=s["seed"], speed=s["speed"], steps=s["steps"], cfg=s["cfg"],
                                     sample_rate=int(s["sample_rate"]), eq=s["eq"])

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
           "--base-directory", str(base), "--extra-model-paths-config", str(yaml)]
    proc = subprocess.Popen(cmd + ([] if args.gpu else ["--cpu"]), cwd=comfy, stdout=log, stderr=subprocess.STDOUT)
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")

        def run(name: str, tag: str, expect_success: bool = True, **overrides) -> dict:
            wf = json.loads(json.dumps(workflows[name]))
            for node in wf.values():
                if "filename_prefix" in node["inputs"]:
                    node["inputs"]["filename_prefix"] = node["inputs"]["filename_prefix"].replace("antalia/", f"ci/{tag}_")
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            try:
                result = queue_client.run(server, wf, None, 600)
            except SystemExit as rejected:  # POST /prompt refused the workflow (validation error)
                result = {"status": "rejected", "wall_seconds": 0, "error": str(rejected)[:300]}
            error = result.get("error") or {}
            print(json.dumps({"run": tag, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}, ensure_ascii=False))
            if (result["status"] == "success") != expect_success:
                failures.append(f"{tag}: expected {'success' if expect_success else 'an error'}, got {result['status']} {error}")
            return result

        def saved(stem: str):
            files = sorted((output / "ci").glob(f"{stem}_*.flac"))
            if len(files) != 1:
                failures.append(f"{stem}: expected one saved .flac, found {[f.name for f in files]}")
                return None, None
            data, rate = sf.read(str(files[0]), dtype="float32", always_2d=True)
            return data[:, 0] if data.shape[1] == 1 else data, rate

        def same(tag: str, got, rate, expected):
            if got is None:
                return
            ok_shape = got.shape == expected.audio.shape
            diff = float(np.abs(got - np.clip(expected.audio, -1, 1)).max()) if ok_shape else None
            print(json.dumps({"compare": tag, "rate": rate, "samples": int(got.shape[0]), "max_diff_in_16bit_steps": None if diff is None else round(diff / STEP, 3)}))
            if rate != expected.sample_rate or diff is None or diff > STEP:
                failures.append(f"{tag}: saved FLAC differs from the official audio ({rate} Hz, diff {diff}, {got.shape} vs {expected.audio.shape})")

        result = run("speak", "speak_a")
        got, rate = saved("speak_a_speech")
        same("speak_a", got, rate, reference("speak"))
        spoken = [v[0] for v in (result.get("text") or {}).values() if v]
        expected_text = Antalia.normalize(workflows["speak"]["2"]["inputs"]["text"])
        if spoken != [expected_text]:
            failures.append(f"spoken text {spoken} != {expected_text!r}")

        run("speak_phone", "phone")
        got, rate = saved("phone_phone")
        same("phone (8 kHz)", got, rate, reference("speak_phone"))

        run("speak", "fp16", **{"1.variant": "fp16"})
        got, rate = saved("fp16_speech")
        same("fp16", got, rate, reference("speak", "fp16"))

        run("speak", "seed9", **{"2.seed": 9})
        got9, _ = saved("seed9_speech")
        same("seed 9", got9, 48000, reference("speak", seed=9))

        run("speak", "speak_b")  # after other jobs: ComfyUI's cache no longer holds speak_a
        a, _ = saved("speak_a_speech")
        b, _ = saved("speak_b_speech")
        if a is not None and b is not None and not np.array_equal(a, b):
            failures.append("the same job gave different audio")
        if a is not None and got9 is not None and a.shape == got9.shape and np.array_equal(a, got9):
            failures.append("seeds 0 and 9 gave the same audio")

        run("speak", "bad_model", expect_success=False, **{"1.model": "../antalia-mini"})
        run("speak", "bad_rate", expect_success=False, **{"2.sample_rate": "44100"})
        run("speak", "empty_text", expect_success=False, **{"2.text": "   "})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

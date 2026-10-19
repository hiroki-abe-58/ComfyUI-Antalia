"""Download Antalia-2 Mini into ComfyUI's models/antalia, and verify it.

    python tools/download_model.py --models-dir <ComfyUI>/models/antalia [--variant fp32|fp16|all] [--verify-only]

Source (pinned revision; Apache-2.0, see the model card):
  https://huggingface.co/cloud0day3/antalia-mini   46 MB for both variants (fp32 30.5 MB, fp16 15.3 MB)
Result:
  models/antalia/antalia-mini/config.json, acoustic(.fp16).safetensors, vocoder(.fp16).safetensors, LICENSE, NOTICE, README.md
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
requirements). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO, REVISION = "cloud0day3/antalia-mini", "8a79528770af2ed5cb6d3b9ac1248d370429f272"
NAME = "antalia-mini"
WEIGHTS = {"fp32": ("acoustic.safetensors", "vocoder.safetensors"), "fp16": ("acoustic.fp16.safetensors", "vocoder.fp16.safetensors")}
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{file name: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name.split("/", 1)[1]] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/antalia folder")
    parser.add_argument("--variant", choices=["all", *WEIGHTS], default="all", help="weight files to fetch (default: both)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    skip = {f for v, names in WEIGHTS.items() if args.variant not in ("all", v) for f in names}
    folder = args.models_dir / NAME
    ok = True
    for name, digest in expected_files().items():
        if name in skip:
            continue
        target = folder / name
        if not args.verify_only and not (target.is_file() and sha256(target) == digest):
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=REPO, filename=name, revision=REVISION, local_dir=str(folder))
        good = target.is_file() and sha256(target) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if target.is_file() else 'MISSING ')} {NAME}/{name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

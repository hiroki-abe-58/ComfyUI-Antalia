"""Antalia-2 Mini for ComfyUI (unofficial integration of https://huggingface.co/cloud0day3/antalia-mini)."""

import logging

try:
    from .antalia_comfy.nodes import comfy_entrypoint  # noqa: F401
except ImportError as e:  # ComfyUI without the V3 node API, or a missing dependency
    logging.error("[Antalia] not loaded: %s", e)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

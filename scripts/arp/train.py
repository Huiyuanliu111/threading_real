#!/usr/bin/env python3
"""Train an ARP checkpoint for the real Threading task.

This stable entry point delegates to the existing Hydra trainer and its
configuration files in ``pushbox/configs``.
W&B defaults to online; pass ``logging.mode=offline`` or ``logging.mode=disabled``
explicitly to override it.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if not any(arg.lstrip("+").split("=", 1)[0] == "logging.mode" for arg in sys.argv[1:]):
        sys.argv.append("logging.mode=online")
    runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "pushbox" / "train.py"),
        run_name="__main__",
    )

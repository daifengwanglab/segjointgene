#!/usr/bin/env python3
"""Verify core dependencies for SegJointGene-CID standalone package."""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "PIL",
    "hydra",
    "omegaconf",
    "pytorch_lightning",
    "tensorboard",
    "sklearn",
    "skimage",
    "scipy",
    "pandas",
    "matplotlib",
    "tqdm",
    "joblib",
]


def main() -> int:
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    print(f"[check_env] python: {sys.executable}")
    print(f"[check_env] version: {sys.version.split()[0]}")
    if missing:
        print("[check_env] missing:", ", ".join(missing))
        print("[check_env] install: python3 -m pip install -r requirements.txt")
        return 1
    print("[check_env] all core packages import OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Training dependency probe: used by ``main.py`` and :mod:`dlbase.train` before Hydra runs.

Import this submodule directly (``from dlbase.runtime_check import ...``); package
:mod:`dlbase` does not eagerly import PyTorch (see :mod:`dlbase` lazy exports).
"""

from __future__ import annotations

import sys
from typing import List, Tuple

_REQUIRED: Tuple[Tuple[str, str], ...] = (
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("PIL", "Pillow"),
    ("cv2", "opencv-python"),
)


def missing_packages() -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for mod, pip in _REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, pip))
    return missing


def print_python_identity() -> None:
    print("[env] python:", sys.executable)
    print("[env] version:", sys.version.split()[0])


def verify_training_runtime(*, context: str = "DLBase") -> None:
    print_python_identity()
    missing = missing_packages()
    if not missing:
        return
    print(
        f"[env] Missing packages in the current environment ({context}).\n"
        "Install into the **same** interpreter you use to run main/train, then retry:\n"
        f"  {sys.executable} -m pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    for mod, pip in missing:
        print(f"  - cannot import `{mod}`  (see requirements: {pip})", file=sys.stderr)
    print(
        "\nTip: run `which python3` and ensure your shell activates the intended venv before "
        "`pip install` and before `python3 main.py`.",
        file=sys.stderr,
    )
    raise SystemExit(1)

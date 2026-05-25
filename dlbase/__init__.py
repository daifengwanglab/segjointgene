"""
DLBase: portable training utilities.

Heavy imports (e.g. PyTorch / Lightning) are **lazy** so ``dlbase.runtime_check`` can run
before optional dependencies are validated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "TaskModule",
    "apply_lightning_reproducibility",
    "verify_training_runtime",
]


def __getattr__(name: str) -> Any:
    if name == "TaskModule":
        from dlbase.registry import TaskModule

        return TaskModule
    if name == "apply_lightning_reproducibility":
        from dlbase.training.seed import apply_lightning_reproducibility

        return apply_lightning_reproducibility
    if name == "verify_training_runtime":
        from dlbase.runtime_check import verify_training_runtime

        return verify_training_runtime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from dlbase.training.seed import apply_lightning_reproducibility
    from dlbase.registry import TaskModule
    from dlbase.runtime_check import verify_training_runtime

"""One-shot import of plugin registration modules (datasets / models / tasks)."""

from __future__ import annotations

_loaded = False


def ensure_plugins_loaded() -> None:
    """Import plugin side-effect modules exactly once per process."""
    global _loaded
    if _loaded:
        return
    import datasets.plugins  # noqa: F401 — registers datasets
    import models.plugins  # noqa: F401
    import tasks.plugins  # noqa: F401

    _loaded = True

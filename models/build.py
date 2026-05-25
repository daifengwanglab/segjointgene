"""Build ``torch.nn.Module`` from Hydra config (plugins registered in ``models.plugins``)."""

from __future__ import annotations

from omegaconf import DictConfig


def construct_model(cfg: DictConfig):
    """Construct weights only (call :func:`build_model` for compatibility checks)."""
    from dlbase.registry import get_model_plugin

    return get_model_plugin(str(cfg.model.name)).construct(cfg)


def build_model(cfg: DictConfig):
    """Assert task×dataset compatibility for this model, then construct the module."""
    from dlbase.registry import get_model_plugin

    plugin = get_model_plugin(str(cfg.model.name))
    plugin.assert_compatible(cfg)
    return plugin.construct(cfg)

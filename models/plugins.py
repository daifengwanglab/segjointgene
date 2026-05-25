"""Register UNet model (standalone Simulation + CID package)."""

from __future__ import annotations

from omegaconf import DictConfig

from dlbase.registry import register_model


def _register_builtin_models() -> None:
    def assert_unet(cfg: DictConfig) -> None:
        from models.unet.model_task_dataset_compatibility import assert_compatible

        assert_compatible(cfg)

    def construct_unet(cfg: DictConfig):
        from models.unet.unet import UNet

        m = cfg.model
        return UNet(
            n_channels=int(m.in_channels),
            n_classes=int(m.num_classes),
            bilinear=bool(m.bilinear),
            base_c=int(m.base_c),
        )

    register_model("unet", assert_unet, construct_unet)


_register_builtin_models()

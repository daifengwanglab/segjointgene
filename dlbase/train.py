"""
Hydra training entry (lives under ``dlbase/``). Prefer ``python main.py`` from repo root.

Direct use::

  python -m dlbase.train
  python -m dlbase.train model=<name> train.max_epochs=50

``config_path`` points at the repository ``conf/`` directory (one level above this file).
"""

from __future__ import annotations

from typing import Optional

from dlbase.runtime_check import verify_training_runtime

verify_training_runtime(context="dlbase.train")

import dlbase.plugin_bootstrap as _plugin_bootstrap

_plugin_bootstrap.ensure_plugins_loaded()

import hydra
from omegaconf import DictConfig, OmegaConf

from dlbase.cfg.config_validate import validate_hydra_cfg


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    validate_hydra_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    from tasks.runner import run_task

    run_task(cfg)
    return None


if __name__ == "__main__":
    main()

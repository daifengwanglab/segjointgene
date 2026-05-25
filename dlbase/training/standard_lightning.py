"""
Reusable Lightning training stack: loggers, checkpoints, Trainer kwargs, TensorBoard helper.

Task packages compose their own ``LightningModule`` and phase logic; this module hosts **engineering**
pieces described in README (experiment leaf cleanup, TensorBoard/W&B loggers, checkpoint filenames,
tqdm/LR wiring helpers). Toggle via ``train.facets`` (see ``dlbase.training.facets``).
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn


def env_suppresses_tensorboard_auto_launch() -> bool:
    """True during pytest or when ``DLBASE_SMOKE`` is set (registry smoke / quick runs without TB)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    v = os.environ.get("DLBASE_SMOKE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def clear_experiment_leaf_dir(cwd: str) -> None:
    """Delete all files/subdirs under the run leaf except ``.hydra`` (Hydra job metadata)."""
    if not os.path.isdir(cwd):
        return
    removed: List[str] = []
    for name in os.listdir(cwd):
        if name == ".hydra":
            continue
        path = os.path.join(cwd, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            removed.append(name)
        except OSError as exc:
            rank_zero_warn(f"Could not remove {path}: {exc}")
    if removed:
        rank_zero_info(
            "Cleared previous artifacts in output dir (sub_path leaf): "
            + ", ".join(sorted(removed))
            + " — kept `.hydra/` for this Hydra run."
        )


def early_stopping_patience_epochs(max_epochs: int) -> int:
    """Same rule as LR warmup in this codebase: ``max(5, int(0.1 * max_epochs))``."""
    return max(5, int(0.1 * int(max_epochs)))


def build_lightning_loggers(cfg: DictConfig) -> List:
    """TensorBoard and/or W&B loggers from ``cfg.logger``."""
    loggers: List = []
    if cfg.logger.tensorboard:
        loggers.append(
            TensorBoardLogger(
                save_dir=os.getcwd(),
                name="tensorboard",
                default_hp_metric=False,
            )
        )
    if cfg.logger.wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger
        except ImportError as e:
            raise ImportError("W&B logging requires: pip install wandb") from e
        loggers.append(
            WandbLogger(
                project=cfg.logger.wandb_project,
                name=cfg.logger.wandb_name,
                entity=cfg.logger.wandb_entity if cfg.logger.wandb_entity else None,
                offline=cfg.logger.wandb_offline,
                log_model=False,
            )
        )
    if not loggers:
        raise ValueError("Enable at least one logger in conf (tensorboard or wandb), or set train.facets.lightning_loggers=false and pass logger=False to Trainer.")
    return loggers


def build_model_checkpoint_callbacks(cfg: DictConfig, dirpath: str) -> Tuple[ModelCheckpoint, List[ModelCheckpoint]]:
    """Monitored top-1 as ``best.ckpt``; optional periodic ``epoch_<epoch>.ckpt``."""
    c = cfg.train.checkpoint
    mc_best = ModelCheckpoint(
        dirpath=dirpath,
        monitor=c.monitor,
        mode=c.mode,
        save_top_k=1,
        save_last=False,
        filename="best",
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )
    extra: List[ModelCheckpoint] = []
    if bool(c.get("keep_intermediate", False)):
        n = max(1, int(c.get("keep_every_n_epochs", 10)))
        extra.append(
            ModelCheckpoint(
                dirpath=dirpath,
                monitor=None,
                save_top_k=-1,
                save_last=False,
                filename="epoch_{epoch}",
                auto_insert_metric_name=False,
                every_n_epochs=n,
            )
        )
    return mc_best, extra


def lightning_trainer_kwargs(
    cfg: DictConfig,
    max_epochs: int,
    callbacks: List,
    loggers: Any,
    *,
    enable_progress_bar: bool,
) -> Dict[str, Any]:
    """Standard ``pl.Trainer`` kwargs shared across task implementations."""
    t = cfg.train
    kw: Dict[str, Any] = {
        "max_epochs": max_epochs,
        "accelerator": t.accelerator,
        "devices": t.devices,
        "logger": loggers,
        "callbacks": callbacks,
        "log_every_n_steps": t.log_every_n_steps,
        "deterministic": t.deterministic,
        "benchmark": t.benchmark,
        "check_val_every_n_epoch": 1,
        "enable_progress_bar": enable_progress_bar,
    }
    if t.gradient_clip_val and t.gradient_clip_val > 0:
        kw["gradient_clip_val"] = t.gradient_clip_val
    if t.precision is not None:
        kw["precision"] = t.precision
    return kw


def _find_free_port(host: str = "127.0.0.1", start: int = 6006, span: int = 64) -> int:
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    return start


def _tensorboard_argv() -> list[str]:
    exe = shutil.which("tensorboard")
    if exe:
        return [exe]
    return [sys.executable, "-m", "tensorboard.main"]


def _wait_tensorboard_http(url: str, timeout_s: float = 25.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:  # noqa: S310
                if 200 <= int(resp.status) < 400:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            time.sleep(0.3)
    return False


@rank_zero_only
def maybe_launch_tensorboard_browser(cwd: str, cfg: DictConfig) -> None:
    """Start TensorBoard on ``cwd/tensorboard``, wait for HTTP, then open browser (rank 0)."""
    if env_suppresses_tensorboard_auto_launch():
        return
    if not cfg.logger.tensorboard:
        return
    if not bool(cfg.logger.get("tensorboard_auto_launch", True)):
        return
    logdir = os.path.join(cwd, "tensorboard")
    if not os.path.isdir(logdir):
        rank_zero_warn(f"TensorBoard log directory not found ({logdir}); skip auto-launch.")
        return
    bind_all = bool(cfg.logger.get("tensorboard_bind_all", False))
    port = _find_free_port()
    prefix = _tensorboard_argv()
    if bind_all:
        cmd = [*prefix, "--logdir", logdir, "--bind_all", "--port", str(port)]
    else:
        cmd = [*prefix, "--logdir", logdir, "--host", "127.0.0.1", "--port", str(port)]
    manual = " ".join(shlex.quote(c) for c in cmd)
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        rank_zero_warn(f"TensorBoard failed to start: {exc}\n  {manual}")
        return
    url = f"http://127.0.0.1:{port}/"
    if not _wait_tensorboard_http(url):
        rank_zero_warn(f"TensorBoard did not respond at {url} (timeout).\n  {manual}")
        return
    try:
        opened = webbrowser.open(url)
    except Exception as exc:  # noqa: BLE001
        rank_zero_warn(f"TensorBoard at {url} — browser open error: {exc}\n  {manual}")
        return
    if opened:
        rank_zero_info(f"TensorBoard {url} (bind_all={bind_all})  |  {manual}")
    else:
        rank_zero_info(f"TensorBoard {url} — browser returned False; open manually  |  {manual}")

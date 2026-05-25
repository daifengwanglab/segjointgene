"""Training helpers for dynamic segmentation: paths, LR, checkpoints, label cache (TensorBoard for metrics)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Tuple

import torch
from omegaconf import DictConfig

from models.build import build_model
from tasks.dynamic_segmentation.training_compat.checkpoint_io import load_checkpoint, resolve_checkpoint_path, save_checkpoint
from tasks.dynamic_segmentation.training_compat.epoch_aggregates import StatHistory


def path_dict_from_cfg(cfg: DictConfig) -> dict[str, Any]:
    root = Path(cfg.paths.root).resolve()
    ds_name = cfg.dataset.name
    exp_ds = root / "experiment" / str(ds_name)
    exp_ds.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    return {
        "root_path": str(root),
        "net_sub_path": cwd,
        "experiment_dataset_path": str(exp_ds),
    }


def step_set_seed(path_dict: dict, net: torch.nn.Module) -> torch.nn.Module:
    p = os.path.join(path_dict["net_sub_path"], "net_seed.bin")
    torch.save(net.state_dict(), p)
    return net


def step_get_optimizer(net: torch.nn.Module, cfg: DictConfig):
    opt = cfg.train.optimizer
    name = str(opt.name)
    lr = float(opt.lr)
    wd = float(opt.weight_decay)
    if name == "Adam":
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    if name == "AdamW":
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
    raise ValueError(f"Unsupported optimizer: {name}")


def set_lr(
    cfg: DictConfig,
    epoch_id: int,
    optimizer: torch.optim.Optimizer,
    warmup_epoch: int = 5,
    min_lr_ratio: float = 0.01,
) -> None:
    all_epoch = int(cfg.train.max_epochs)
    peak_lr = float(cfg.train.optimizer.lr)
    min_lr = peak_lr * min_lr_ratio
    if all_epoch <= warmup_epoch:
        current_lr = peak_lr * (epoch_id + 1) / max(1, all_epoch)
    else:
        if epoch_id < warmup_epoch:
            current_lr = peak_lr * (epoch_id + 1) / warmup_epoch
        else:
            denom = all_epoch - warmup_epoch - 1
            if denom <= 0:
                current_lr = min_lr
            else:
                t = epoch_id - warmup_epoch
                progress = t / denom
                current_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    print("current epoch: ", epoch_id)
    print("current lr: ", current_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr


def step_plot_classify(path_dict, stat_history, args, prefix: str = "") -> None:
    """Reserved: primary metrics go to TensorBoard."""
    del path_dict, stat_history, args, prefix


def step_plot_loss(path_dict, stat_history, args, plot_name: str, prefix: str) -> None:
    """Reserved: primary metrics go to TensorBoard."""
    del path_dict, stat_history, args, plot_name, prefix


def step_print_epoch(epoch_id: int, stat_history: StatHistory, start_time: float, end_time: float) -> None:
    print_content = "Epoch: " + str(epoch_id) + "\n"
    for key in stat_history.stat.keys():
        print_content += key + ": " + str(stat_history.stat[key][epoch_id]) + "\n"
    print_content += "time: " + str(end_time - start_time) + "\n"
    print(print_content)


def step_save_ckpt(
    path_dict: dict,
    epoch_id: int,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    stat_history: StatHistory,
    args,
) -> None:
    path = os.path.join(path_dict["net_sub_path"], f"epoch_{epoch_id}.ckpt")
    extra = stat_history.to_checkpoint_extra()
    if getattr(args, "save_space_trick", False):
        if epoch_id % int(getattr(args, "save_space_trick_epoch_num", 1)) == 0:
            save_checkpoint(path, epoch_id, net, optimizer, extra=extra)
    else:
        save_checkpoint(path, epoch_id, net, optimizer, extra=extra)


def step_save_stat(path_dict: dict, epoch_id: int, stat_history: StatHistory, step_suffix: str, args) -> None:
    del path_dict, epoch_id, stat_history, step_suffix, args


def step_save_summary(path_dict: dict, stat_history: StatHistory, step_suffix: str, args) -> None:
    del path_dict, stat_history, step_suffix, args


def step_save_label_cache(path_dict: dict, epoch_id: int, train_set, test_set) -> None:
    save_path = os.path.join(path_dict["net_sub_path"], f"labels_cache_{epoch_id}.pt")
    state = {
        "train_dynamic_class": train_set.dynamic_class_labels,
        "train_dynamic_inst": train_set.dynamic_instance_labels,
        "test_dynamic_class": test_set.dynamic_class_labels,
        "test_dynamic_inst": test_set.dynamic_instance_labels,
    }
    torch.save(state, save_path)
    print(f"[IO] Label Cache saved to {save_path}")


def step_load_label_cache(path_dict: dict, start_epoch: int, train_set, test_set) -> None:
    if start_epoch < 1:
        print(f"[IO] Start epoch is {start_epoch}, skipping label load (using raw labels).")
        return
    target_epoch = start_epoch
    load_path = os.path.join(path_dict["net_sub_path"], f"labels_cache_{target_epoch}.pt")
    if os.path.exists(load_path):
        print(f"[IO] Loading Label Cache from {load_path}...")
        state = torch.load(load_path, weights_only=False)
        train_set.dynamic_class_labels = state["train_dynamic_class"]
        train_set.dynamic_instance_labels = state["train_dynamic_inst"]
        test_set.dynamic_class_labels = state["test_dynamic_class"]
        test_set.dynamic_instance_labels = state["test_dynamic_inst"]
        print(
            f"[IO] Label Cache restored. Train keys: {len(train_set.dynamic_class_labels)}, "
            f"Test keys: {len(test_set.dynamic_class_labels)}"
        )
    else:
        print(f"[IO] Warning: Label file {load_path} not found! Starting with raw labels.")


def step_save_emb_cache(path_dict: dict, tag: str, train_emb: dict, test_emb: dict) -> None:
    save_path = os.path.join(path_dict["net_sub_path"], f"emb_cache_{tag}.pt")
    torch.save({"train": train_emb, "test": test_emb}, save_path)
    print(f"[IO] Embedding Cache saved to {save_path}")


def step_load_ckpt(
    path_dict: dict,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args,
    if_load: bool,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, StatHistory, int]:
    stat_history = StatHistory()
    if not if_load:
        return net, optimizer, stat_history, 0
    net_sub_path = path_dict["net_sub_path"]
    ckpt_path = resolve_checkpoint_path(net_sub_path, int(args.ckpt_load_epoch))
    if ckpt_path is None:
        legacy = os.path.join(net_sub_path, "net_" + str(args.ckpt_load_epoch) + ".ckpt")
        ckpt_path = legacy if os.path.isfile(legacy) else None
    print("Load from:", ckpt_path)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError("the path of net does not exsits.")
    ep, extra = load_checkpoint(ckpt_path, net, optimizer)
    stat_history = StatHistory.from_checkpoint_extra(extra)
    start_epoch = ep if ep is not None else int(args.ckpt_load_epoch)
    return net, optimizer, stat_history, start_epoch


def build_net(cfg: DictConfig) -> torch.nn.Module:
    return build_model(cfg)

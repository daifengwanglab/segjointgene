"""
Hydra-driven dynamic segmentation with CID attribution (WMB / CA1 patches + UNet).

Extends the base ``dynamic_segmentation`` pipeline with:
  - CID (or IG) attribution per batch when ``epoch >= attr_epoch``
  - Attribution-aware ``update_label`` (expand_k, attr_top_ratio)
  - Attribution-aware global stitcher (grid-level heatmaps)
  - End-of-epoch attribution CSV + heatmap per group_col
  - Explicit VRAM management (``del attr; torch.cuda.empty_cache()``)
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn

from datasets.datamodule import DLBaseDataModule
from dlbase.training.seed import apply_lightning_reproducibility
from dlbase.training.standard_lightning import maybe_launch_tensorboard_browser
from tasks.dynamic_segmentation.config_utils import hydra_cfg_to_args
from tasks.dynamic_segmentation.get_gene_celltype import get_gene_celltype
from tasks.dynamic_segmentation.loss import (
    HybridLoss,
    accumulate_global_miou_hist,
    compute_class_weight_from_loader,
    compute_mIOU,
    global_dice_from_accumulated_hist,
    global_miou_from_accumulated_hist,
)
from tasks.dynamic_segmentation.step_compat import (
    build_net,
    path_dict_from_cfg,
    set_lr,
    step_get_optimizer,
    step_load_ckpt,
    step_load_label_cache,
    step_plot_classify,
    step_plot_loss,
    step_print_epoch,
    step_save_ckpt,
    step_save_label_cache,
    step_save_stat,
    step_save_summary,
    step_set_seed,
)
from tasks.dynamic_segmentation.batch_unpack import unpack_seg_batch
from tasks.dynamic_segmentation.gene_miou import (
    compute_simulation_gene_miou_bar_from_stitcher_by_gr,
    compute_simulation_gene_miou_from_stitcher,
)
from tasks.dynamic_segmentation.simulation_eval_bars import simulation_bar_index_from_gr_gc
from tasks.cell_morphology import compute_morphology_from_instance_maps
from tasks.dynamic_segmentation.update_label import update_label
from tasks.dynamic_segmentation.dataset_preprocess import maybe_run_dataset_preprocess
from tasks.dynamic_segmentation_CID.pipeline_checks import run_final_pre_training_checks
from tasks.dynamic_segmentation_CID.compute_CID import compute_CID
from tasks.dynamic_segmentation_CID.global_stitcher_CID import GlobalStitchingEvaluatorCID
from tasks.dynamic_segmentation.summary_support import maybe_write_summary_log
from tasks.dynamic_segmentation.training_compat.epoch_aggregates import EpochBatchAggregator, StatHistory
from tasks.dynamic_segmentation.training_compat.experiment_paths import tensorboard_log_dir
from tasks.dynamic_segmentation.training_compat.tensorboard_metrics import TensorBoardEpochLogger
from tasks.dynamic_segmentation.visualize import (
    TestStitchCache,
    render_panels_from_payload,
    save_payload,
)
from tasks.dynamic_segmentation.visualize.metrics_payload import SegEvalPayload


def _infer_test_group_row(*eval_sets) -> int:
    from collections import Counter

    gr_counter: Counter[int] = Counter()
    for es in eval_sets:
        for path in es.file_list:
            parts = os.path.basename(path).split(".")[0].split("_")
            gr_counter[int(parts[1])] += 1
    if not gr_counter:
        raise RuntimeError("Cannot infer test_group_row: eval sets are empty.")
    dominant = gr_counter.most_common(1)[0][0]
    print(f"[run_CID] inferred test_group_row={dominant} from eval sets ({dict(gr_counter)})")
    return dominant


def _clear_experiment_leaf_dir(cwd: str) -> None:
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
            "Cleared previous artifacts in output dir: "
            + ", ".join(sorted(removed))
            + " — kept `.hydra/`."
        )


def _compute_attr_and_pool(net, image, target_gene, target_celltype, args):
    """Run CID and return ``(net, attr_5d, visualize_attr, cid_time_stats)``."""
    net, attr, cid_time_stats = compute_CID(
        net,
        image,
        target_gene,
        target_celltype,
        n_steps=int(args.CID_n_steps),
        lr=float(args.CID_lr),
        lambda_param=float(args.CID_lambda_param),
        beta=float(args.CID_beta),
        grid_size=int(args.CID_grid_size),
        noise_num=int(args.CID_noise_num),
        gene_chunk_size=int(args.CID_gene_chunk_size),
        return_spatial=True,
    )
    attr = attr.detach()
    B, Cc, Cg, H, W = attr.shape
    gs = int(args.CID_grid_size)
    visualize_attr = F.avg_pool2d(
        attr.float().view(B, Cc * Cg, H, W),
        kernel_size=gs,
        stride=gs,
    ).view(B, Cc, Cg, H // gs, W // gs).cpu()
    return net, attr, visualize_attr, cid_time_stats


def _append_cid_timing_record(records, *, epoch_id: int, split: str, batch_id: int, cid_time_stats: dict | None):
    if not cid_time_stats:
        return
    row = {
        "epoch": int(epoch_id),
        "split": str(split),
        "batch_id": int(batch_id),
    }
    row["cid_time_total_sec"] = float(cid_time_stats.get("00_Total", 0.0))
    for k, v in cid_time_stats.items():
        row[f"cid_time__{k}"] = float(v)
    records.append(row)


def _print_test_timing_brief(records, *, epoch_id: int) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    part = df[(df["epoch"] == int(epoch_id)) & (df["split"] == "test")]
    if part.empty:
        return
    step_cols = [c for c in part.columns if c.startswith("cid_time__")]
    if not step_cols:
        return
    means = part[step_cols].mean(axis=0).sort_values(ascending=False)
    top = means.head(3)
    top_txt = ", ".join(f"{k.replace('cid_time__', '')}: {float(v):.4f}s" for k, v in top.items())
    print(
        f"[Epoch {epoch_id}] CID timing (test) — batches={len(part)}, "
        f"mean total={float(part['cid_time_total_sec'].mean()):.4f}s; top steps: {top_txt}"
    )


def _save_attr_csv_and_heatmap(
    stitcher: GlobalStitchingEvaluatorCID,
    epoch_id: int,
    vis_base: str,
    target_gene_names,
    target_celltype_names,
    *,
    figure_dpi: float,
):
    """End-of-epoch attribution CSV + heatmap per group_col."""
    for gc in range(stitcher.n_group_col):
        current_attr = stitcher.global_attr[gc]
        attr_score = current_attr.mean(dim=(2, 3))
        gene_score = attr_score.mean(dim=0)
        cell_score = attr_score.mean(dim=1)
        gene_order = torch.argsort(gene_score, descending=True)
        gene_names_sorted = [target_gene_names[j] for j in gene_order.tolist()]
        cell_order = torch.argsort(cell_score, descending=True)
        cell_names_sorted = [target_celltype_names[i] for i in cell_order.tolist()]
        global_attr_sorted = current_attr[cell_order][:, gene_order]

        non_zero_mask = (global_attr_sorted != 0).float()
        sum_attr = torch.sum(global_attr_sorted, dim=(2, 3))
        count_attr = torch.sum(non_zero_mask, dim=(2, 3))
        global_attr_sorted = sum_attr / count_attr.clamp(min=1e-9)

        vis_dir = os.path.join(vis_base, "visualize", f"epoch_{epoch_id}")
        os.makedirs(vis_dir, exist_ok=True)

        df = pd.DataFrame(
            global_attr_sorted.cpu().numpy(),
            index=cell_names_sorted,
            columns=gene_names_sorted,
        )
        csv_path = os.path.join(vis_dir, f"gene_celltype_attribution_gc{gc}.csv")
        df.to_csv(csv_path)

        min_val = global_attr_sorted.min()
        max_val = global_attr_sorted.max()
        global_attr_normalized = (global_attr_sorted - min_val) / (max_val - min_val + 1e-8)

        fig, ax = plt.subplots(
            figsize=(0.4 * len(gene_names_sorted) + 4, 0.4 * len(cell_names_sorted) + 4)
        )
        im = ax.imshow(
            global_attr_normalized.cpu().numpy(),
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_xticks(range(len(gene_names_sorted)))
        ax.set_xticklabels(gene_names_sorted, rotation=90, fontsize=8)
        ax.set_yticks(range(len(cell_names_sorted)))
        ax.set_yticklabels(cell_names_sorted, fontsize=8)
        ax.set_xlabel("Gene", fontsize=10)
        ax.set_ylabel("Cell Type", fontsize=10)
        ax.set_title(f"Gene–Celltype Attribution (Epoch {epoch_id}, Col {gc})", fontsize=14)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Normalized Attribution Strength", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(vis_dir, f"gene_celltype_attribution_gc{gc}.png"),
            dpi=float(figure_dpi),
        )
        plt.close(fig)


def run_dynamic_segmentation_CID_training(cfg: DictConfig) -> None:
    _clear_experiment_leaf_dir(os.getcwd())
    train_start = datetime.now()
    apply_lightning_reproducibility(int(cfg.seed), workers=True)
    maybe_run_dataset_preprocess(cfg)
    run_final_pre_training_checks(cfg)
    args = hydra_cfg_to_args(cfg)
    rank_zero_info(
        f"[run] predict_epoch={int(args.predict_epoch)} (max_epochs={int(cfg.train.max_epochs)}; "
        "default max(5, int(0.1*max_epochs)) when dynamic_seg.predict_epoch is unset)"
    )
    path_dict = path_dict_from_cfg(cfg)

    dm = DLBaseDataModule(cfg)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    train_set = dm.train_set
    val_set = dm.val_set
    test_set = dm.test_set

    target_gene, target_celltype, target_gene_names, target_celltype_names = get_gene_celltype(
        str(cfg.dataset.data_path), args, train_loader
    )

    test_group_row = _infer_test_group_row(val_set, test_set)

    ds_name = str(cfg.dataset.name)
    if ds_name == "CA1":
        n_global_col = 4
    else:
        n_global_col = 1

    num_update_iterations = 10

    net = build_net(cfg).cuda()
    net = step_set_seed(path_dict, net)
    optimizer = step_get_optimizer(net, cfg)
    num_classes = int(args.n_celltype)

    tb_log_dir = tensorboard_log_dir(path_dict["net_sub_path"])
    tb_logger = TensorBoardEpochLogger(tb_log_dir)

    net, optimizer, stat_history, start_epoch = step_load_ckpt(
        path_dict, net, optimizer, args, if_load=bool(args.if_load_ckpt)
    )
    if args.if_load_ckpt:
        step_load_label_cache(path_dict, start_epoch, train_set, test_set)

    step_suffix = str(cfg.task.name)
    best_val_loss = float("inf")
    best_epoch = -1
    vis_dir = path_dict["net_sub_path"]
    _vis_every_cid = max(1, int(getattr(args, "vis_save_every", 10)))
    _max_epochs_cid = int(cfg.train.max_epochs)

    def _cid_should_save_visualization(eid: int) -> bool:
        return eid == 0 or eid % _vis_every_cid == 0 or eid == _max_epochs_cid

    latest_eval_payload: SegEvalPayload | None = None

    attr_method = str(getattr(args, "attr_method", "CID"))
    attr_epoch = int(getattr(args, "attr_epoch", 0))
    cid_timing_records: list[dict] = []

    try:
        for epoch_id in range(start_epoch, int(cfg.train.max_epochs) + 1):
            set_lr(cfg, epoch_id, optimizer)
            start_time = time.time()
            batch_agg = EpochBatchAggregator()

            class_weight = compute_class_weight_from_loader(train_loader, num_classes=num_classes)
            criterion = HybridLoss(
                class_weight=class_weight,
                num_classes=num_classes,
                lambda_ce=float(args.lambda_ce),
                lambda_dice=float(args.lambda_dice),
            ).cuda()

            if attr_method == "none":
                run_attr = False
                save_epoch_num = 10
            else:
                run_attr = epoch_id >= attr_epoch
                save_epoch_num = 1

            stitcher = GlobalStitchingEvaluatorCID(
                test_set=test_set,
                train_set=train_set,
                patch_size=int(args.patch_size),
                pixel_distance=int(args.pixel_distance),
                target_gene=target_gene,
                target_celltypes=target_celltype,
                target_gene_names=target_gene_names,
                target_celltype_names=target_celltype_names,
                if_attr=run_attr,
                grid_size=int(args.CID_grid_size),
                test_group_row=test_group_row,
                n_group_col=n_global_col,
                val_set=val_set,
            )

            print("run_attr", run_attr)
            print("if_attr", stitcher.if_attr)

            test_viz_cache = TestStitchCache(
                n_group_col=int(stitcher.n_group_col),
                h_global=int(stitcher.H_global),
                w_global=int(stitcher.W_global),
                patch_size=int(args.patch_size),
                test_group_row=int(test_group_row),
            )

            merged_val_test = val_set is test_set

            train_global_hist = torch.zeros(
                num_classes, num_classes, device="cuda", dtype=torch.float32
            )
            val_global_hist = torch.zeros(
                num_classes, num_classes, device="cuda", dtype=torch.float32
            )
            test_global_hist = torch.zeros(
                num_classes, num_classes, device="cuda", dtype=torch.float32
            )
            _sim_ds = str(cfg.dataset.name) == "Simulation"
            test_hist_sim_bar = None
            if _sim_ds:
                test_hist_sim_bar = [
                    torch.zeros(num_classes, num_classes, device="cuda", dtype=torch.float32)
                    for _ in range(4)
                ]

            # ================= Train Loop =================
            net.train()
            for batch_id, batch_data in enumerate(train_loader):
                print("batch:", batch_id)
                batch_data, ground_truth_instance = unpack_seg_batch(batch_data)
                (
                    image,
                    label,
                    instance_label,
                    spots,
                    dapi,
                    idx,
                    group_rows,
                    group_cols,
                    rows,
                    cols,
                    fixed_label,
                    fixed_inst,
                    ground_truth,
                ) = batch_data
                del dapi, fixed_label

                image, label, fixed_inst, ground_truth = (
                    image.cuda(),
                    label.cuda().long(),
                    fixed_inst.cuda().long(),
                    ground_truth.cuda().long(),
                )
                if ground_truth_instance is not None:
                    ground_truth_instance = ground_truth_instance.cuda().long()
                spots = spots.cuda()
                instance_label = instance_label.cuda()

                output = net(image)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                ce_l, dice_l = criterion(output, label)
                loss = ce_l + dice_l
                mIOU = compute_mIOU(output, label, num_classes)

                if epoch_id != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # --- Attribution ---
                attr = None
                visualize_attr = None
                cid_time_stats = None
                if run_attr and attr_method == "CID":
                    net, attr, visualize_attr, cid_time_stats = _compute_attr_and_pool(
                        net, image, target_gene, target_celltype, args
                    )
                    _append_cid_timing_record(
                        cid_timing_records,
                        epoch_id=epoch_id,
                        split="train",
                        batch_id=batch_id,
                        cid_time_stats=cid_time_stats,
                    )

                # --- Label Update ---
                if epoch_id <= int(args.predict_epoch):
                    new_label, new_inst = label.detach().clone(), instance_label.detach().clone()
                else:
                    for _ in range(num_update_iterations):
                        label, instance_label = update_label(
                            label,
                            instance_label,
                            (fixed_inst > 0),
                            output,
                            epoch_id,
                            batch_id,
                            expand_k=int(args.expand_k),
                            threshold=float(args.prediction_threshold),
                            attributions=attr,
                            target_celltype=target_celltype,
                            attr_top_ratio=float(getattr(args, "attr_top_ratio", 0.1)),
                        )
                    new_inst = instance_label
                    new_label = label

                if attr is not None:
                    del attr

                accumulate_global_miou_hist(new_label, ground_truth, train_global_hist, num_classes)

                train_set.update_label_cache(idx, new_label, new_inst)

                _pi = new_inst.cpu() if ground_truth_instance is not None else None
                _gi = ground_truth_instance.cpu() if ground_truth_instance is not None else None
                _sim_inst = str(cfg.dataset.name) == "Simulation" and ground_truth_instance is not None
                stitcher.update(
                    new_label,
                    spots,
                    rows,
                    cols,
                    group_rows,
                    group_cols,
                    attributions=visualize_attr,
                    test_group_row=test_group_row,
                    pred_insts=_pi,
                    gt_insts=_gi,
                    inst_split="train" if _sim_inst else None,
                )

                bs = image.size(0)
                ce_item = float(ce_l.detach().cpu())
                dice_item = float(dice_l.detach().cpu())
                tot_item = ce_item + dice_item
                pix_acc = float((torch.argmax(output, 1) == label).float().mean().item())
                lam_d = float(getattr(args, "lambda_dice", 1.0))
                current_soft_dice = (
                    1.0 - (dice_item / lam_d) if lam_d > 1e-12 else 0.0
                )
                batch_agg.update("train_ce", ce_item, bs)
                batch_agg.update("train_total", tot_item, bs)
                batch_agg.update("train_dice_loss", dice_item, bs)
                batch_agg.update("train_acc", pix_acc, bs)
                batch_agg.update("train_current_mIOU", float(mIOU), bs)
                batch_agg.update("train_current_dice", current_soft_dice, bs)

                torch.cuda.empty_cache()

            # ================= Val Loop =================
            net.eval()
            if not merged_val_test:
                for batch_id, batch_data in enumerate(val_loader):
                    print("batch:", batch_id)
                    batch_data, ground_truth_instance = unpack_seg_batch(batch_data)
                    (
                        image,
                        label,
                        instance_label,
                        spots,
                        dapi,
                        idx,
                        group_rows,
                        group_cols,
                        rows,
                        cols,
                        fixed_label,
                        fixed_inst,
                        ground_truth,
                    ) = batch_data
                    del dapi, fixed_label

                    image, label, fixed_inst, ground_truth = (
                        image.cuda(),
                        label.cuda().long(),
                        fixed_inst.cuda().long(),
                        ground_truth.cuda().long(),
                    )
                    if ground_truth_instance is not None:
                        ground_truth_instance = ground_truth_instance.cuda().long()
                    spots = spots.cuda()
                    instance_label = instance_label.cuda()

                    with torch.no_grad():
                        output = net(image)
                        if isinstance(output, (tuple, list)):
                            output = output[0]
                        ce_l, dice_l = criterion(output, label)
                        mIOU = compute_mIOU(output, label, num_classes)

                    attr = None
                    visualize_attr = None
                    cid_time_stats = None
                    if run_attr and attr_method == "CID":
                        net, attr, visualize_attr, cid_time_stats = _compute_attr_and_pool(
                            net, image, target_gene, target_celltype, args
                        )
                        _append_cid_timing_record(
                            cid_timing_records,
                            epoch_id=epoch_id,
                            split="val",
                            batch_id=batch_id,
                            cid_time_stats=cid_time_stats,
                        )

                    if epoch_id <= int(args.predict_epoch):
                        new_label, new_inst = label.detach().clone(), instance_label.detach().clone()
                    else:
                        for _ in range(num_update_iterations):
                            label, instance_label = update_label(
                                label,
                                instance_label,
                                (fixed_inst > 0),
                                output,
                                epoch_id,
                                batch_id,
                                expand_k=int(args.expand_k),
                                threshold=float(args.prediction_threshold),
                                attributions=attr,
                                target_celltype=target_celltype,
                                attr_top_ratio=float(getattr(args, "attr_top_ratio", 0.1)),
                            )
                        new_inst = instance_label
                        new_label = label

                    if attr is not None:
                        del attr

                    accumulate_global_miou_hist(new_label, ground_truth, val_global_hist, num_classes)
                    val_set.update_label_cache(idx, new_label, new_inst)

                    _pi = new_inst.cpu() if ground_truth_instance is not None else None
                    _gi = ground_truth_instance.cpu() if ground_truth_instance is not None else None
                    _sim_inst = str(cfg.dataset.name) == "Simulation" and ground_truth_instance is not None
                    stitcher.update(
                        new_label,
                        spots,
                        rows,
                        cols,
                        group_rows,
                        group_cols,
                        attributions=visualize_attr,
                        test_group_row=test_group_row,
                        pred_insts=_pi,
                        gt_insts=_gi,
                        inst_split="val" if _sim_inst else None,
                    )

                    bs = image.size(0)
                    ce_item = float(ce_l.detach().cpu())
                    dice_item = float(dice_l.detach().cpu())
                    tot_item = ce_item + dice_item
                    pix_acc = float((torch.argmax(output, 1) == label).float().mean().item())
                    lam_d = float(getattr(args, "lambda_dice", 1.0))
                    current_soft_dice = (
                        1.0 - (dice_item / lam_d) if lam_d > 1e-12 else 0.0
                    )
                    batch_agg.update("val_ce", ce_item, bs)
                    batch_agg.update("val_total", tot_item, bs)
                    batch_agg.update("val_dice_loss", dice_item, bs)
                    batch_agg.update("val_acc", pix_acc, bs)
                    batch_agg.update("val_current_mIOU", float(mIOU), bs)
                    batch_agg.update("val_current_dice", current_soft_dice, bs)

                    torch.cuda.empty_cache()

            # ================= Test Loop =================
            net.eval()
            for batch_id, batch_data in enumerate(test_loader):
                print("batch:", batch_id)
                batch_data, ground_truth_instance = unpack_seg_batch(batch_data)
                (
                    image,
                    label,
                    instance_label,
                    spots,
                    dapi,
                    idx,
                    group_rows,
                    group_cols,
                    rows,
                    cols,
                    fixed_label,
                    fixed_inst,
                    ground_truth,
                ) = batch_data
                del dapi, fixed_label

                image, label, fixed_inst, ground_truth = (
                    image.cuda(),
                    label.cuda().long(),
                    fixed_inst.cuda().long(),
                    ground_truth.cuda().long(),
                )
                if ground_truth_instance is not None:
                    ground_truth_instance = ground_truth_instance.cuda().long()
                spots = spots.cuda()
                instance_label = instance_label.cuda()

                with torch.no_grad():
                    output = net(image)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    ce_l, dice_l = criterion(output, label)
                    mIOU = compute_mIOU(output, label, num_classes)

                attr = None
                visualize_attr = None
                cid_time_stats = None
                if run_attr and attr_method == "CID":
                    net, attr, visualize_attr, cid_time_stats = _compute_attr_and_pool(
                        net, image, target_gene, target_celltype, args
                    )
                    _append_cid_timing_record(
                        cid_timing_records,
                        epoch_id=epoch_id,
                        split="test",
                        batch_id=batch_id,
                        cid_time_stats=cid_time_stats,
                    )

                if epoch_id <= int(args.predict_epoch):
                    new_label, new_inst = label.detach().clone(), instance_label.detach().clone()
                else:
                    for _ in range(num_update_iterations):
                        label, instance_label = update_label(
                            label,
                            instance_label,
                            (fixed_inst > 0),
                            output,
                            epoch_id,
                            batch_id,
                            expand_k=int(args.expand_k),
                            threshold=float(args.prediction_threshold),
                            attributions=attr,
                            target_celltype=target_celltype,
                            attr_top_ratio=float(getattr(args, "attr_top_ratio", 0.1)),
                        )
                    new_inst = instance_label
                    new_label = label

                if attr is not None:
                    del attr

                accumulate_global_miou_hist(new_label, ground_truth, test_global_hist, num_classes)
                if _sim_ds and test_hist_sim_bar is not None:
                    gr_np = group_rows.detach().cpu().numpy()
                    gc_np = group_cols.detach().cpu().numpy()
                    for b in range(int(new_label.shape[0])):
                        bi = simulation_bar_index_from_gr_gc(int(gr_np[b]), int(gc_np[b]))
                        if bi is not None:
                            accumulate_global_miou_hist(
                                new_label[b : b + 1],
                                ground_truth[b : b + 1],
                                test_hist_sim_bar[bi],
                                num_classes,
                            )
                test_set.update_label_cache(idx, new_label, new_inst)

                _pi = new_inst.cpu() if ground_truth_instance is not None else None
                _gi = ground_truth_instance.cpu() if ground_truth_instance is not None else None
                _sim_inst = str(cfg.dataset.name) == "Simulation" and ground_truth_instance is not None
                stitcher.update(
                    new_label,
                    spots,
                    rows,
                    cols,
                    group_rows,
                    group_cols,
                    attributions=visualize_attr,
                    test_group_row=test_group_row,
                    pred_insts=_pi,
                    gt_insts=_gi,
                    inst_split="test" if _sim_inst else None,
                )
                test_viz_cache.update(
                    labels=new_label,
                    spots=spots,
                    pred_insts=new_inst,
                    rows=rows,
                    cols=cols,
                    group_rows=group_rows,
                    group_cols=group_cols,
                )

                bs = image.size(0)
                ce_item = float(ce_l.detach().cpu())
                dice_item = float(dice_l.detach().cpu())
                tot_item = ce_item + dice_item
                pix_acc = float((torch.argmax(output, 1) == label).float().mean().item())
                lam_d = float(getattr(args, "lambda_dice", 1.0))
                current_soft_dice = (
                    1.0 - (dice_item / lam_d) if lam_d > 1e-12 else 0.0
                )
                batch_agg.update("test_ce", ce_item, bs)
                batch_agg.update("test_total", tot_item, bs)
                batch_agg.update("test_dice_loss", dice_item, bs)
                batch_agg.update("test_acc", pix_acc, bs)
                batch_agg.update("test_current_mIOU", float(mIOU), bs)
                batch_agg.update("test_current_dice", current_soft_dice, bs)

                if merged_val_test:
                    accumulate_global_miou_hist(new_label, ground_truth, val_global_hist, num_classes)
                    batch_agg.update("val_ce", ce_item, bs)
                    batch_agg.update("val_total", tot_item, bs)
                    batch_agg.update("val_dice_loss", dice_item, bs)
                    batch_agg.update("val_acc", pix_acc, bs)
                    batch_agg.update("val_current_mIOU", float(mIOU), bs)
                    batch_agg.update("val_current_dice", current_soft_dice, bs)

                torch.cuda.empty_cache()

            # --- End Epoch ---
            cell_calling_score = stitcher.compute_score()
            print(f"\n[Epoch {epoch_id}] Cell calling score: {cell_calling_score}")

            if stitcher.global_pred_inst.any():
                morpho = compute_morphology_from_instance_maps(
                    stitcher.global_pred_inst.numpy()
                )
            else:
                morpho = {"cell_area_mean": 0.0, "cell_convexity_mean": 0.0,
                          "cell_elongation_mean": 0.0, "cell_count": 0.0}
            print(
                f"[Epoch {epoch_id}] Morphology — area: {morpho['cell_area_mean']:.1f}, "
                f"convexity: {morpho['cell_convexity_mean']:.4f}, "
                f"elongation: {morpho['cell_elongation_mean']:.4f}, "
                f"count: {morpho['cell_count']:.0f}"
            )

            train_global_mIOU = global_miou_from_accumulated_hist(train_global_hist, num_classes)
            val_global_mIOU = global_miou_from_accumulated_hist(val_global_hist, num_classes)
            test_global_mIOU = global_miou_from_accumulated_hist(test_global_hist, num_classes)
            train_global_dice = global_dice_from_accumulated_hist(train_global_hist, num_classes)
            val_global_dice = global_dice_from_accumulated_hist(val_global_hist, num_classes)
            test_global_dice = global_dice_from_accumulated_hist(test_global_hist, num_classes)

            gene_miou_train = 0.0
            gene_miou_val = 0.0
            gene_miou_test = 0.0
            # global_* / Gene mIoU vs GT: stitched pseudo-labels (new_label / new_inst), not raw argmax(logits).
            if str(cfg.dataset.name) == "Simulation":
                gene_miou_train = compute_simulation_gene_miou_from_stitcher(
                    stitcher,
                    str(cfg.paths.data_dir),
                    test_group_row,
                    split="train",
                    global_scale=float(cfg.dataset.preprocess.global_scale),
                )
                if merged_val_test:
                    gene_miou_val = 0.0
                else:
                    gene_miou_val = compute_simulation_gene_miou_from_stitcher(
                        stitcher,
                        str(cfg.paths.data_dir),
                        test_group_row,
                        split="val",
                        global_scale=float(cfg.dataset.preprocess.global_scale),
                    )
                gene_miou_test = compute_simulation_gene_miou_from_stitcher(
                    stitcher,
                    str(cfg.paths.data_dir),
                    test_group_row,
                    split="test",
                    global_scale=float(cfg.dataset.preprocess.global_scale),
                )
                if merged_val_test:
                    gene_miou_val = gene_miou_test
                print(
                    f"[Epoch {epoch_id}] Gene mIoU (Simulation, mean over matched cells) — "
                    f"train: {gene_miou_train:.4f}, val: {gene_miou_val:.4f}, test: {gene_miou_test:.4f}"
                )

            sim_image_miou_bar_np = None
            sim_gene_miou_bar_np = None
            if _sim_ds and test_hist_sim_bar is not None:
                sim_image_miou_bar_np = np.array(
                    [
                        global_miou_from_accumulated_hist(test_hist_sim_bar[i], num_classes)
                        for i in range(4)
                    ],
                    dtype=np.float64,
                )
                sim_gene_miou_bar_np = compute_simulation_gene_miou_bar_from_stitcher_by_gr(
                    stitcher,
                    str(cfg.paths.data_dir),
                    global_scale=float(cfg.dataset.preprocess.global_scale),
                    show_progress=False,
                )

            print(
                f"[Epoch {epoch_id}] global mIOU (GT>0) - Train: {train_global_mIOU:.4f}, "
                f"Val: {val_global_mIOU:.4f}, Test: {test_global_mIOU:.4f} | global Dice - Train: {train_global_dice:.4f}, "
                f"Val: {val_global_dice:.4f}, Test: {test_global_dice:.4f}"
            )
            _print_test_timing_brief(cid_timing_records, epoch_id=epoch_id)

            av = batch_agg.averages()
            row = {
                "train_loss": av.get("train_ce", 0.0),
                "val_loss": av.get("val_ce", 0.0),
                "test_loss": av.get("test_ce", 0.0),
                "train_acc": av.get("train_acc", 0.0),
                "val_acc": av.get("val_acc", 0.0),
                "test_acc": av.get("test_acc", 0.0),
                "train_current_dice": av.get("train_current_dice", 0.0),
                "val_current_dice": av.get("val_current_dice", 0.0),
                "test_current_dice": av.get("test_current_dice", 0.0),
                "current_mIOU_train": av.get("train_current_mIOU", 0.0),
                "current_mIOU_val": av.get("val_current_mIOU", 0.0),
                "current_mIOU_test": av.get("test_current_mIOU", 0.0),
                "train_global_dice": train_global_dice,
                "val_global_dice": val_global_dice,
                "test_global_dice": test_global_dice,
                "global_mIOU_train": train_global_mIOU,
                "global_mIOU_val": val_global_mIOU,
                "global_mIOU_test": test_global_mIOU,
                "train_gene_mIoU": gene_miou_train,
                "val_gene_mIoU": gene_miou_val,
                "test_gene_mIoU": gene_miou_test,
                "cell_calling": cell_calling_score,
                "cell_area": morpho["cell_area_mean"],
                "cell_convexity": morpho["cell_convexity_mean"],
                "cell_elongation": morpho["cell_elongation_mean"],
                "cell_count": morpho["cell_count"],
            }
            latest_eval_payload = test_viz_cache.build_payload(
                epoch_id=epoch_id,
                epoch_tag=f"epoch_{epoch_id}",
                image_miou_test=test_global_mIOU,
                gene_miou_test=gene_miou_test,
                method_name="SegJointGene",
                sim_image_miou_bar=sim_image_miou_bar_np,
                sim_gene_miou_bar=sim_gene_miou_bar_np,
            )
            stat_history.append_epoch(row, epoch_id)
            stat_history.stat_check()

            tb_logger.log_epoch(
                epoch_id,
                {
                    "train/loss": av.get("train_total", 0.0),
                    "train/ce": av.get("train_ce", 0.0),
                    "train/dice_loss": av.get("train_dice_loss", 0.0),
                    "val/loss": av.get("val_total", 0.0),
                    "val/ce": av.get("val_ce", 0.0),
                    "val/dice_loss": av.get("val_dice_loss", 0.0),
                    "test/loss": av.get("test_total", 0.0),
                    "test/ce": av.get("test_ce", 0.0),
                    "test/dice_loss": av.get("test_dice_loss", 0.0),
                    "train/current_mIOU": av.get("train_current_mIOU", 0.0),
                    "val/current_mIOU": av.get("val_current_mIOU", 0.0),
                    "test/current_mIOU": av.get("test_current_mIOU", 0.0),
                    "train/pixel_acc": av.get("train_acc", 0.0),
                    "val/pixel_acc": av.get("val_acc", 0.0),
                    "test/pixel_acc": av.get("test_acc", 0.0),
                    "train/global_mIOU": train_global_mIOU,
                    "val/global_mIOU": val_global_mIOU,
                    "test/global_mIOU": test_global_mIOU,
                    "train/current_dice": av.get("train_current_dice", 0.0),
                    "val/current_dice": av.get("val_current_dice", 0.0),
                    "test/current_dice": av.get("test_current_dice", 0.0),
                    "train/global_dice": train_global_dice,
                    "val/global_dice": val_global_dice,
                    "test/global_dice": test_global_dice,
                    "train/gene_mIoU": gene_miou_train,
                    "val/gene_mIoU": gene_miou_val,
                    "test/gene_mIoU": gene_miou_test,
                    "cell_calling": cell_calling_score,
                    "cell_area": morpho["cell_area_mean"],
                    "cell_convexity": morpho["cell_convexity_mean"],
                    "cell_elongation": morpho["cell_elongation_mean"],
                    "cell_count": morpho["cell_count"],
                },
            )

            step_plot_classify(path_dict, stat_history, args)
            step_plot_loss(path_dict, stat_history, args, "current_dice", prefix="current_dice")
            step_plot_loss(path_dict, stat_history, args, "current_mIOU", prefix="current_mIOU")
            step_plot_loss(path_dict, stat_history, args, "global_dice", prefix="global_dice")
            step_plot_loss(path_dict, stat_history, args, "global_mIOU", prefix="global_mIOU")
            step_print_epoch(epoch_id, stat_history, start_time, time.time())
            step_save_stat(path_dict, epoch_id, stat_history, step_suffix, args)

            # --- Save policy ---
            if epoch_id % save_epoch_num == 0:
                step_save_ckpt(path_dict, epoch_id, net, optimizer, stat_history, args)
                step_save_label_cache(path_dict, epoch_id, train_set, test_set)
            if _cid_should_save_visualization(epoch_id):
                _fd = float(getattr(args, "figure_dpi", 600))
                stitcher.save_visualize(vis_dir, epoch_id, args=args, scale=1.0, save_dpi=_fd)
                if latest_eval_payload is not None:
                    epoch_vis_dir = os.path.join(vis_dir, "visualize", f"epoch_{epoch_id}")
                    save_payload(epoch_vis_dir, latest_eval_payload)
                    panel_paths = render_panels_from_payload(
                        latest_eval_payload, epoch_vis_dir, figure_dpi=_fd
                    )
                    names = ", ".join(os.path.basename(p) for p in panel_paths)
                    print(f"[IO] Evaluation panels saved ({names}) under {epoch_vis_dir}")
                if run_attr:
                    _save_attr_csv_and_heatmap(
                        stitcher,
                        epoch_id,
                        vis_dir,
                        target_gene_names,
                        target_celltype_names,
                        figure_dpi=_fd,
                    )

            cur_val_loss = av.get("val_total", float("inf"))
            if cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                best_epoch = epoch_id
                print(f"[best] New best at epoch {epoch_id} (val_loss={best_val_loss:.4f})")

    finally:
        tb_logger.close()

    if cid_timing_records:
        timing_df = pd.DataFrame(cid_timing_records)
        timing_path = os.path.join(path_dict["net_sub_path"], "cid_timing_batches.csv")
        timing_df.to_csv(timing_path, index=False)
        print(f"[IO] CID batch timing saved: {timing_path}")

        step_cols = [c for c in timing_df.columns if c.startswith("cid_time__")]
        summary_rows = []
        for (eid, split), part in timing_df.groupby(["epoch", "split"], sort=True):
            row = {"epoch": int(eid), "split": str(split), "n_batches": int(len(part))}
            row["cid_time_total_sec__sum"] = float(part["cid_time_total_sec"].sum())
            row["cid_time_total_sec__mean"] = float(part["cid_time_total_sec"].mean())
            for c in step_cols:
                row[f"{c}__sum"] = float(part[c].sum())
                row[f"{c}__mean"] = float(part[c].mean())
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows).sort_values(["epoch", "split"])
        summary_path = os.path.join(path_dict["net_sub_path"], "cid_timing_epoch_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"[IO] CID epoch timing summary saved: {summary_path}")

    print(f"[done] Best epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    step_save_summary(path_dict, stat_history, step_suffix, args)

    maybe_write_summary_log(
        cfg=cfg,
        dm=dm,
        stat_history=stat_history,
        net=net,
        train_start=train_start,
        train_end=datetime.now(),
        start_epoch=start_epoch,
        best_epoch=best_epoch,
        best_test_loss=best_val_loss,
        args=args,
        num_classes=num_classes,
    )
    maybe_launch_tensorboard_browser(os.getcwd(), cfg)

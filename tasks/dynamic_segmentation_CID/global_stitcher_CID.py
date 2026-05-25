"""
Attribution-aware global stitcher for dynamic_segmentation_CID.

Extends the base stitcher with:
  - ``grid_size`` for grid-level global dimensions (Hg_global, Wg_global)
  - ``if_attr`` flag that allocates ``global_attr`` tensor
  - ``update()`` accepts ``attributions`` kwarg for grid-level stitching
  - ``save_visualize()`` renders attribution heatmaps + ``rescale_global_attributes``
"""

from __future__ import annotations

import os
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def rescale_global_attributes(global_attr: torch.Tensor) -> torch.Tensor:
    """
    Global percentile normalization + per-location gene-centering.

    Steps:
      1) Use global p50 as 0 and p90 as 1 for affine normalization.
      2) For each (celltype, y, x), compute mean across genes.
      3) Subtract that mean from each gene map and clamp negatives to 0.
    """
    arr = global_attr.numpy().reshape(-1)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return torch.zeros_like(global_attr)

    p50, p90 = np.percentile(arr[valid], [50.0, 90.0])
    denom = float(max(p90 - p50, 1e-8))
    print(f"[CID vis] percentile normalization: p50={float(p50):.6f}, p90={float(p90):.6f}")

    normalized = (global_attr - float(p50)) / denom
    gene_mean = normalized.mean(dim=1, keepdim=True)
    rescaled_attr = normalized - gene_mean
    rescaled_attr = torch.clamp(rescaled_attr, min=0.0)
    return rescaled_attr


class GlobalStitchingEvaluatorCID:
    """Patch-to-global stitcher with optional attribution heatmap support."""

    def __init__(
        self,
        test_set,
        train_set,
        patch_size: int,
        pixel_distance: int,
        target_gene,
        target_celltypes,
        target_gene_names,
        target_celltype_names,
        if_attr: bool,
        grid_size: int,
        test_group_row: int,
        n_group_col: int,
        val_set=None,
    ):
        self.patch_size = int(patch_size)
        self.grid_size = int(grid_size)
        self.patch_size_grid = self.patch_size // self.grid_size

        self.d = int(pixel_distance)
        self.kernel_size = 2 * self.d + 1
        self.epsilon = 1e-6

        self.target_gene = target_gene
        self.target_celltypes = target_celltypes
        self.target_gene_names = target_gene_names
        self.target_celltype_names = target_celltype_names
        self.n_gene = len(target_gene)
        self.n_celltype = len(target_celltypes)
        self.if_attr = if_attr

        self.test_group_row = test_group_row
        self.n_group_col = n_group_col

        max_row = 0
        max_col = 0
        max_gc = 0
        max_gr = 0
        all_files = list(test_set.file_list) + list(train_set.file_list)
        if val_set is not None:
            all_files = all_files + list(val_set.file_list)
        for path in all_files:
            basename = os.path.basename(path)
            if not basename.startswith("p_"):
                continue
            try:
                parts = basename.split(".")[0].split("_")
                gr = int(parts[1])
                gc = int(parts[2])
                r, c = int(parts[3]), int(parts[4])
                max_gr = max(max_gr, gr)
                max_gc = max(max_gc, gc)
                max_row = max(max_row, r)
                max_col = max(max_col, c)
            except (IndexError, ValueError):
                continue

        self.n_group_row = max(1, int(max_gr) + 1)

        inferred_gc = max_gc + 1
        if inferred_gc > self.n_group_col:
            self.n_group_col = inferred_gc

        self.rows_count = max_row + 1
        self.cols_count = max_col + 1

        self.H_global = self.rows_count * self.patch_size
        self.W_global = self.cols_count * self.patch_size

        self.Hg_global = self.rows_count * self.patch_size_grid
        self.Wg_global = self.cols_count * self.patch_size_grid

        print("\n[Stitcher CID Init]")
        print(f"  Pixel Global Size : {self.W_global} x {self.H_global}")
        print(f"  Grid  Global Size : {self.Wg_global} x {self.Hg_global}")
        print(f"  Grid Config       : {self.n_celltype} CellTypes x {self.n_gene} Genes")
        print(
            f"  Group Config      : Row {self.test_group_row}, Cols {self.n_group_col} "
            f"(n_group_row={self.n_group_row})"
        )

        self.global_mask = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_spots = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.float32
        )
        self.global_pred_inst = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_gt_inst = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_pred_inst_train = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_gt_inst_train = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_pred_inst_test = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_gt_inst_test = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_pred_inst_val = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_gt_inst_val = torch.zeros(
            (self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_pred_inst_test_by_gr = torch.zeros(
            (self.n_group_row, self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.global_gt_inst_test_by_gr = torch.zeros(
            (self.n_group_row, self.n_group_col, self.H_global, self.W_global), dtype=torch.long
        )
        self.occurrence_map = torch.zeros(
            (self.n_group_col, self.rows_count, self.cols_count), dtype=torch.int32
        )

        if self.if_attr:
            self.global_attr = torch.zeros(
                (self.n_group_col, self.n_celltype, self.n_gene, self.Hg_global, self.Wg_global),
                dtype=torch.float32,
            )

    def update(
        self,
        labels,
        spots,
        rows,
        cols,
        group_rows,
        group_cols,
        attributions=None,
        test_group_row=None,
        pred_insts=None,
        gt_insts=None,
        inst_split: Optional[Literal["train", "val", "test"]] = None,
    ):
        tgt_row = test_group_row if test_group_row is not None else self.test_group_row

        labels = labels.detach().cpu()
        spots = spots.detach().cpu()
        rows = rows.detach().cpu().numpy()
        cols = cols.detach().cpu().numpy()
        group_rows = group_rows.detach().cpu().numpy()
        group_cols = group_cols.detach().cpu().numpy()

        if attributions is not None:
            attributions = attributions.detach().cpu()
        if pred_insts is not None:
            pred_insts = pred_insts.detach().cpu()
        if gt_insts is not None:
            gt_insts = gt_insts.detach().cpu()

        B = labels.shape[0]

        for i in range(B):
            gr, gc = int(group_rows[i]), int(group_cols[i])
            if gc >= self.n_group_col:
                continue

            r, c = int(rows[i]), int(cols[i])
            y_start = r * self.patch_size
            x_start = c * self.patch_size
            y_end = y_start + self.patch_size
            x_end = x_start + self.patch_size

            if gr == tgt_row:
                self.occurrence_map[gc, r, c] += 1

                self.global_mask[gc, y_start:y_end, x_start:x_end] = labels[i]
                self.global_spots[gc, y_start:y_end, x_start:x_end] = spots[i]
                if pred_insts is not None:
                    if inst_split == "train":
                        self.global_pred_inst_train[gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                    elif inst_split == "val":
                        self.global_pred_inst_val[gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                    elif inst_split == "test":
                        self.global_pred_inst_test[gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                        self.global_pred_inst[gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                    else:
                        self.global_pred_inst[gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                if gt_insts is not None:
                    if inst_split == "train":
                        self.global_gt_inst_train[gc, y_start:y_end, x_start:x_end] = gt_insts[i]
                    elif inst_split == "val":
                        self.global_gt_inst_val[gc, y_start:y_end, x_start:x_end] = gt_insts[i]
                    elif inst_split == "test":
                        self.global_gt_inst_test[gc, y_start:y_end, x_start:x_end] = gt_insts[i]
                        self.global_gt_inst[gc, y_start:y_end, x_start:x_end] = gt_insts[i]
                    else:
                        self.global_gt_inst[gc, y_start:y_end, x_start:x_end] = gt_insts[i]

                if attributions is not None:
                    gy_start = r * self.patch_size_grid
                    gx_start = c * self.patch_size_grid
                    gy_end = gy_start + self.patch_size_grid
                    gx_end = gx_start + self.patch_size_grid

                    self.global_attr[gc, :, :, gy_start:gy_end, gx_start:gx_end] = attributions[i]

            if (
                pred_insts is not None
                and gt_insts is not None
                and inst_split == "test"
                and gr < self.n_group_row
            ):
                self.global_pred_inst_test_by_gr[gr, gc, y_start:y_end, x_start:x_end] = pred_insts[i]
                self.global_gt_inst_test_by_gr[gr, gc, y_start:y_end, x_start:x_end] = gt_insts[i]

    @torch.no_grad()
    def compute_score(self):
        scores = []
        for gc in range(self.n_group_col):
            g_mask_gpu = self.global_mask[gc].cuda()
            g_spots_gpu = self.global_spots[gc].cuda()

            mask_in = g_mask_gpu > 0
            if mask_in.sum() == 0:
                scores.append(0.0)
                continue

            mask_in_float = mask_in.float().unsqueeze(0).unsqueeze(0)
            mask_dilated = (
                F.max_pool2d(mask_in_float, kernel_size=self.kernel_size, stride=1, padding=self.d).squeeze() > 0
            )
            mask_margin = mask_dilated & (~mask_in)

            total_gin = (g_spots_gpu * mask_in).sum().item()
            total_gout = (g_spots_gpu * mask_margin).sum().item()

            del g_mask_gpu, g_spots_gpu, mask_in, mask_dilated, mask_margin
            torch.cuda.empty_cache()

            if total_gout == 0:
                scores.append(0.0)
            else:
                scores.append(total_gin / (total_gout + self.epsilon))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"Final Score (Avg over {self.n_group_col} cols): {avg_score:.4f}")
        return avg_score

    def save_visualize(
        self,
        save_root: str,
        epoch_id: int,
        args,
        scale: float = 1.0,
        *,
        save_dpi: float | None = None,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dpi = (
            float(save_dpi)
            if save_dpi is not None
            else float(getattr(args, "figure_dpi", 600))
        )

        vis_dir = os.path.join(save_root, "visualize", f"epoch_{epoch_id}")
        os.makedirs(vis_dir, exist_ok=True)

        for gc in range(self.n_group_col):
            mask_np = self.global_mask[gc].numpy().astype(np.float32)
            if mask_np.sum() == 0:
                continue
            target_w = int(self.W_global * scale)
            target_h = int(self.H_global * scale)
            if target_w < 1 or target_h < 1:
                continue
            mask_resized = cv2.resize(
                mask_np,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            plt.figure(figsize=(12, 12), dpi=dpi)
            plt.imshow(mask_resized, cmap="viridis", interpolation="nearest")
            plt.axis("off")
            plt.title(f"Global Mask (Epoch {epoch_id}, Col {gc})")
            plt.savefig(
                os.path.join(vis_dir, f"global_mask_epoch_{epoch_id}_gc{gc}.png"),
                dpi=dpi,
            )
            plt.close()

            if not self.if_attr:
                continue

            current_attr = self.global_attr[gc]
            np_path = os.path.join(vis_dir, f"attr_{gc}.npy")
            np.save(np_path, current_attr.cpu().numpy() if torch.is_tensor(current_attr) else current_attr)

            attr_score = current_attr.mean(dim=(2, 3))
            gene_score = attr_score.mean(dim=0)
            cell_score = attr_score.mean(dim=1)
            K = min(5, self.n_celltype, self.n_gene)
            gene_order = torch.argsort(gene_score, descending=True)[:K]
            gene_names_sorted = [self.target_gene_names[j] for j in gene_order.tolist()]

            if hasattr(args, "datasets_name") and str(args.datasets_name).startswith("WMB"):
                target_cell_names = [
                    "NP-CT-L6b Glut",
                    "IT-ET Glut",
                    "OPC-Oligo",
                    "CTX-CGE GABA",
                    "CTX-MGE GABA",
                ]
                cell_order_list = [
                    self.target_celltype_names.index(name)
                    for name in target_cell_names
                    if name in self.target_celltype_names
                ]
                if len(cell_order_list) == 0:
                    cell_order = torch.argsort(cell_score, descending=True)[:K]
                    cell_names_sorted = [self.target_celltype_names[i] for i in cell_order.tolist()]
                else:
                    K = min(K, len(cell_order_list))
                    cell_order = torch.tensor(cell_order_list[:K], dtype=torch.long, device=current_attr.device)
                    cell_names_sorted = [self.target_celltype_names[i] for i in cell_order_list[:K]]
            else:
                cell_order = torch.argsort(cell_score, descending=True)[:K]
                cell_names_sorted = [self.target_celltype_names[i] for i in cell_order.tolist()]

            gene_order = gene_order[:K]
            gene_names_sorted = gene_names_sorted[:K]
            global_attr_sorted = current_attr[cell_order][:, gene_order]
            global_attr_sorted = rescale_global_attributes(global_attr_sorted)

            fig, axes = plt.subplots(K, K, figsize=(4 * K, 4 * K), squeeze=False, dpi=dpi)
            plt.subplots_adjust(wspace=0.05, hspace=0.1)

            vmax = float(global_attr_sorted.max().item()) if global_attr_sorted.numel() > 0 else 1.0
            if vmax <= 0:
                vmax = 1.0

            im = None
            for i in range(K):
                for j in range(K):
                    ax = axes[i, j]
                    im = ax.imshow(
                        global_attr_sorted[i, j].numpy(),
                        cmap="plasma",
                        vmin=0,
                        vmax=vmax,
                    )
                    ax.axis("off")
                    if i == 0:
                        ax.set_title(f"Gene {gene_names_sorted[j]}", fontsize=10, fontweight="bold")
                    if j == 0:
                        ax.text(
                            -0.1,
                            0.5,
                            f"Cell {cell_names_sorted[i]}",
                            transform=ax.transAxes,
                            rotation=90,
                            va="center",
                            ha="right",
                            fontsize=10,
                            fontweight="bold",
                        )

            cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
            if im is not None:
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label("Percentile-normalized Strength", fontsize=10)

            plt.suptitle(
                f"Attribution (Top-{K}×Top-{K}): Epoch {epoch_id} Col {gc}",
                fontsize=16,
            )
            plt.savefig(
                os.path.join(vis_dir, f"CID_epoch_{epoch_id}_gc{gc}.png"),
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()

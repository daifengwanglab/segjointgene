"""Test-only stitched cache for visualize payload generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from tasks.cell_morphology import calculate_cell_metrics
from tasks.dynamic_segmentation.visualize.metrics_payload import SegEvalPayload


def _to_numpy_int(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.int64)
    return np.asarray(x, dtype=np.int64)


@dataclass
class TestStitchCache:
    """Accumulate stitched test-only label/spot/instance maps for one epoch."""

    n_group_col: int
    h_global: int
    w_global: int
    patch_size: int
    test_group_row: int

    def __post_init__(self) -> None:
        self.test_mask = torch.zeros((self.n_group_col, self.h_global, self.w_global), dtype=torch.long)
        self.test_spots = torch.zeros((self.n_group_col, self.h_global, self.w_global), dtype=torch.float32)
        self.test_pred_inst = torch.zeros((self.n_group_col, self.h_global, self.w_global), dtype=torch.long)

    def update(
        self,
        labels: torch.Tensor,
        spots: torch.Tensor,
        pred_insts: torch.Tensor,
        rows,
        cols,
        group_rows,
        group_cols,
    ) -> None:
        labels_cpu = labels.detach().cpu()
        spots_cpu = spots.detach().cpu()
        pred_insts_cpu = pred_insts.detach().cpu()
        rows_np = _to_numpy_int(rows)
        cols_np = _to_numpy_int(cols)
        group_rows_np = _to_numpy_int(group_rows)
        group_cols_np = _to_numpy_int(group_cols)

        bsz = labels_cpu.shape[0]
        for i in range(bsz):
            gr = int(group_rows_np[i])
            gc = int(group_cols_np[i])
            if gr != int(self.test_group_row):
                continue
            if gc < 0 or gc >= int(self.n_group_col):
                continue

            r = int(rows_np[i])
            c = int(cols_np[i])
            y0 = r * int(self.patch_size)
            x0 = c * int(self.patch_size)
            y1 = y0 + int(self.patch_size)
            x1 = x0 + int(self.patch_size)
            self.test_mask[gc, y0:y1, x0:x1] = labels_cpu[i]
            self.test_spots[gc, y0:y1, x0:x1] = spots_cpu[i]
            self.test_pred_inst[gc, y0:y1, x0:x1] = pred_insts_cpu[i]

    @torch.no_grad()
    def compute_cell_calling_score(self, pixel_distance: int, epsilon: float = 1e-6) -> float:
        d = int(pixel_distance)
        k = 2 * d + 1
        scores = []
        for gc in range(int(self.n_group_col)):
            g_mask = self.test_mask[gc]
            g_spots = self.test_spots[gc]
            mask_in = g_mask > 0
            if int(mask_in.sum().item()) == 0:
                scores.append(0.0)
                continue
            mask_in_float = mask_in.float().unsqueeze(0).unsqueeze(0)
            mask_dilated = F.max_pool2d(mask_in_float, kernel_size=k, stride=1, padding=d).squeeze() > 0
            mask_margin = mask_dilated & (~mask_in)
            total_gin = float((g_spots * mask_in).sum().item())
            total_gout = float((g_spots * mask_margin).sum().item())
            if total_gout == 0.0:
                scores.append(0.0)
            else:
                scores.append(total_gin / (total_gout + float(epsilon)))
        return float(sum(scores) / len(scores)) if scores else 0.0

    def compute_cell_calling_scores(self, distances: Iterable[int] = (3, 5, 7)) -> Dict[int, float]:
        return {int(d): self.compute_cell_calling_score(int(d)) for d in distances}

    def _collect_morphology_distributions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        areas = []
        elongations = []
        convexities = []
        pred_np = self.test_pred_inst.numpy()
        for gc in range(pred_np.shape[0]):
            inst_map = pred_np[gc]
            if not np.any(inst_map):
                continue
            for cell_id in np.unique(inst_map):
                if int(cell_id) == 0:
                    continue
                cell_binary = (inst_map == cell_id).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(cell_binary, connectivity=8)
                for comp_id in range(1, int(num_labels)):
                    comp = (labels == comp_id).astype(np.uint8)
                    if int(comp.sum()) < 4:
                        continue
                    a, c, e = calculate_cell_metrics(comp)
                    areas.append(float(a))
                    convexities.append(float(c))
                    elongations.append(float(e))
        return (
            np.asarray(areas, dtype=np.float64),
            np.asarray(elongations, dtype=np.float64),
            np.asarray(convexities, dtype=np.float64),
        )

    def build_payload(
        self,
        epoch_id: int,
        epoch_tag: str,
        image_miou_test: float,
        gene_miou_test: float,
        method_name: str = "SegJointGene",
        *,
        sim_image_miou_bar: np.ndarray | None = None,
        sim_gene_miou_bar: np.ndarray | None = None,
    ) -> SegEvalPayload:
        cell_calling_scores = self.compute_cell_calling_scores((3, 5, 7))
        area_vals, elong_vals, convex_vals = self._collect_morphology_distributions()
        return SegEvalPayload(
            method_name=str(method_name),
            epoch_tag=str(epoch_tag),
            epoch_id=int(epoch_id),
            image_miou_test=float(image_miou_test),
            gene_miou_test=float(gene_miou_test),
            cell_calling_scores=cell_calling_scores,
            cell_area_values=area_vals,
            cell_elongation_values=elong_vals,
            cell_convexity_values=convex_vals,
            sim_image_miou_bar=(
                np.asarray(sim_image_miou_bar, dtype=np.float64).reshape(4)
                if sim_image_miou_bar is not None
                else None
            ),
            sim_gene_miou_bar=(
                np.asarray(sim_gene_miou_bar, dtype=np.float64).reshape(4)
                if sim_gene_miou_bar is not None
                else None
            ),
        )

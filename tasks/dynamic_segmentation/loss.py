"""Hybrid CE + Dice and metrics for dynamic segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weight_from_loader(loader, num_classes, device="cuda"):
    counts = torch.zeros(num_classes).to(device)
    max_batches = 50
    loader_iter = iter(loader)
    with torch.no_grad():
        for i in range(max_batches):
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                break
            _, label, *_ = batch_data
            label = label.to(device).view(-1)
            b_counts = torch.bincount(label, minlength=num_classes)
            counts += b_counts[:num_classes].float()
    del loader_iter
    counts = counts.clamp(min=1.0)
    total = counts.sum()
    weights = total / counts
    weights = weights / weights.mean()
    weights = weights.clamp(max=10.0)
    weights = weights.clamp(min=0.1)
    return weights


class HybridLoss(nn.Module):
    def __init__(self, class_weight=None, num_classes=10, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.ce = nn.CrossEntropyLoss(weight=self.class_weight)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        smooth = 1.0
        dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice_score_fg = dice_score[:, 1:]
        dice_loss = 1.0 - dice_score_fg.mean()
        return self.lambda_ce * ce_loss, self.lambda_dice * dice_loss


def miou_mean_foreground_from_hist(hist: torch.Tensor, num_classes: int) -> float:
    """Mean IoU over foreground classes (1..C-1) where union > 0. ``hist[gt, pred]`` counts."""
    hist = hist.float()
    intersection = torch.diag(hist)
    union = hist.sum(dim=1) + hist.sum(dim=0) - intersection
    valid_mask = union > 0
    iou = torch.zeros_like(intersection)
    iou[valid_mask] = intersection[valid_mask] / union[valid_mask]
    if num_classes > 1:
        iou_fg = iou[1:]
        valid_fg = valid_mask[1:]
        if valid_fg.sum() > 0:
            return float(iou_fg[valid_fg].mean().item())
        return 0.0
    if valid_mask.sum() > 0:
        return float(iou[valid_mask].mean().item())
    return 0.0


def compute_current_mIOU(output, label, num_classes: int) -> float:
    """Current mIOU: argmax(logits) vs **dynamic training label** on all pixels (batch)."""
    pred = torch.argmax(output, dim=1).flatten()
    label = label.flatten()
    hist = torch.bincount(
        num_classes * label + pred,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    return miou_mean_foreground_from_hist(hist, num_classes)


# Backward-compatible name
compute_mIOU = compute_current_mIOU


def accumulate_global_miou_hist(
    pred_label: torch.Tensor,
    ground_truth: torch.Tensor,
    hist_acc: torch.Tensor,
    num_classes: int,
) -> None:
    """Add confusion counts for pixels with ``ground_truth > 0`` into ``hist_acc`` [gt, pred].

    Pass the **current predicted class map** used for training/eval stitching (dynamic pseudo-label
    ``new_label``), not raw ``argmax(logits)``, so global mIoU / Dice match stitched visualization vs GT.
    """
    valid = ground_truth > 0
    if not torch.any(valid):
        return
    p = pred_label[valid].long().clamp(0, num_classes - 1)
    g = ground_truth[valid].long().clamp(0, num_classes - 1)
    idx = num_classes * g + p
    bc = torch.bincount(idx, minlength=num_classes * num_classes)
    hist_acc += bc.view(num_classes, num_classes).to(dtype=hist_acc.dtype, device=hist_acc.device)


def global_miou_from_accumulated_hist(hist: torch.Tensor, num_classes: int) -> float:
    """Global mIOU: mean foreground IoU from a full-epoch confusion matrix (GT>0 pixels only)."""
    if hist.sum() <= 0:
        return 0.0
    return miou_mean_foreground_from_hist(hist, num_classes)


def dice_mean_foreground_from_hist(hist: torch.Tensor, num_classes: int) -> float:
    """Mean class-wise Dice over foreground classes (1..C-1) with union support, ``hist[gt, pred]``."""
    hist = hist.float()
    diag = torch.diag(hist)
    sum_gt = hist.sum(dim=1)
    sum_pred = hist.sum(dim=0)
    denom = sum_gt + sum_pred
    dice_c = torch.zeros_like(diag)
    ok = denom > 0
    dice_c[ok] = (2.0 * diag[ok]) / denom[ok]
    if num_classes > 1:
        dice_fg = dice_c[1:]
        valid_fg = denom[1:] > 0
        if valid_fg.sum() > 0:
            return float(dice_fg[valid_fg].mean().item())
        return 0.0
    if ok.sum() > 0:
        return float(dice_c[ok].mean().item())
    return 0.0


def global_dice_from_accumulated_hist(hist: torch.Tensor, num_classes: int) -> float:
    """Global Dice: mean foreground Dice from the same accumulated hist as ``global_mIOU`` (pred vs GT, GT>0)."""
    if hist.sum() <= 0:
        return 0.0
    return dice_mean_foreground_from_hist(hist, num_classes)

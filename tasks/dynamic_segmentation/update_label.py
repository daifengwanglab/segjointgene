"""
Sophisticated label update with territory expansion, morphological cleanup,
attribution-aware filtering, and hole filling.

When ``attributions`` is None (default), the attribution filtering step is
skipped and the function behaves as a prediction-only updater -- backward
compatible with the previous simple version.
"""

import numpy as np
import scipy.ndimage as nd
import torch
import torch.nn.functional as F
from skimage.measure import regionprops, label as skimage_label


def balanced_dilate_id(tensor_id, k):
    """Randomized rank-based balanced territory expansion for instance IDs."""
    if k == 0:
        return tensor_id.clone()

    device = tensor_id.device
    curr_mask = tensor_id.clone()

    if curr_mask.dim() == 2:
        curr_mask = curr_mask.unsqueeze(0)

    unique_ids = torch.unique(curr_mask)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        return tensor_id.clone()

    max_id = int(unique_ids.max().item())

    for i in range(k):
        num_ids = len(unique_ids)
        random_ranks = torch.randperm(num_ids, device=device) + 1

        id_to_rank = torch.zeros(max_id + 1, dtype=torch.long, device=device)
        id_to_rank[unique_ids] = random_ranks

        rank_to_id = torch.zeros(num_ids + 1, dtype=torch.long, device=device)
        rank_to_id[random_ranks] = unique_ids

        rank_map = id_to_rank[curr_mask]
        rank_map_float = rank_map.double()

        use_square = (i % 2 == 0)
        if use_square:
            dilated_rank = F.max_pool2d(rank_map_float.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        else:
            dilated_v = F.max_pool2d(rank_map_float.unsqueeze(0), kernel_size=(3, 1), stride=1, padding=(1, 0)).squeeze(0)
            dilated_h = F.max_pool2d(rank_map_float.unsqueeze(0), kernel_size=(1, 3), stride=1, padding=(0, 1)).squeeze(0)
            dilated_rank = torch.max(dilated_v, dilated_h)

        is_background = (curr_mask == 0)
        new_ids = rank_to_id[dilated_rank.long()]
        curr_mask[is_background] = new_ids[is_background]

    if tensor_id.dim() == 2:
        return curr_mask.squeeze(0)
    return curr_mask


def cleanup_islands(instance_label, core_mask, max_iters=100):
    """Keep only connected components that touch core seed pixels."""
    device = instance_label.device
    is_3d = instance_label.dim() == 3
    if not is_3d:
        instance_label = instance_label.unsqueeze(0)
        core_mask = core_mask.unsqueeze(0)

    B = instance_label.shape[0]
    inst_np = instance_label.detach().cpu().numpy()
    core_np = core_mask.detach().cpu().numpy()

    final_label_np = np.zeros_like(inst_np)

    for b in range(B):
        labeled_components = skimage_label(inst_np[b], connectivity=2)
        core_hits = labeled_components[core_np[b]]
        valid_component_ids = np.unique(core_hits)
        valid_component_ids = valid_component_ids[valid_component_ids > 0]
        mask_valid = np.isin(labeled_components, valid_component_ids)
        final_label_np[b][mask_valid] = inst_np[b][mask_valid]

    res = torch.from_numpy(final_label_np).to(device)
    if not is_3d:
        res = res.squeeze(0)
    return res


def fill_instance(instance_label):
    """Fill single-neighbor holes inside instance regions."""
    device = instance_label.device
    label_np = instance_label.detach().cpu().numpy()

    if label_np.ndim == 3:
        for b in range(label_np.shape[0]):
            binary_mask = label_np[b] > 0
            filled_mask = nd.binary_fill_holes(binary_mask)
            holes_mask = filled_mask & (~binary_mask)

            if not holes_mask.any():
                continue

            labeled_holes = skimage_label(holes_mask)
            for prop in regionprops(labeled_holes):
                curr_hole_mask = (labeled_holes == prop.label)
                dilated = nd.binary_dilation(curr_hole_mask, iterations=1)
                neighbor_area = dilated & (~curr_hole_mask)

                neighbor_ids = np.unique(label_np[b][neighbor_area])
                neighbor_ids = neighbor_ids[neighbor_ids > 0]

                if len(neighbor_ids) == 1:
                    label_np[b][curr_hole_mask] = neighbor_ids[0]
        return torch.from_numpy(label_np).to(device)

    else:
        binary_mask = label_np > 0
        filled_mask = nd.binary_fill_holes(binary_mask)
        holes_mask = filled_mask & (~binary_mask)
        if not holes_mask.any():
            return instance_label
        labeled_holes = skimage_label(holes_mask)
        for prop in regionprops(labeled_holes):
            curr_hole_mask = (labeled_holes == prop.label)
            dilated = nd.binary_dilation(curr_hole_mask, iterations=1)
            neighbor_area = dilated & (~curr_hole_mask)
            neighbor_ids = np.unique(label_np[neighbor_area])
            neighbor_ids = neighbor_ids[neighbor_ids > 0]
            if len(neighbor_ids) == 1:
                label_np[curr_hole_mask] = neighbor_ids[0]
        return torch.from_numpy(label_np).to(device)


def update_label(class_label, instance_label, core_mask, output, epoch_id, batch_id,
                 expand_k=2, threshold=0.2, attributions=None, target_celltype=None, attr_top_ratio=0.1):
    """
    10-step label update pipeline:
      0. GlobalID  1. Expand  2. Anchor  3. AttrFilter  4-5. GapFill
      6. Cleanup   7. Revert  8. FillHole  9-10. MapBack + RestoreCore

    When *attributions* is ``None`` (the default), Step 3 is skipped and the
    function is backward-compatible with the old prediction-only updater.
    """
    B, H, W = instance_label.shape
    device = class_label.device
    instance_label = instance_label.to(device)
    core_mask = core_mask.to(device)

    probs = F.softmax(output, dim=1)
    conf, prediction = torch.max(probs, dim=1)
    non_core_zone = (~core_mask)

    # Step 0: Generate global unique IDs across batch
    max_ids = instance_label.view(B, -1).max(dim=1).values
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    if B > 1:
        offsets[1:] = torch.cumsum(max_ids[:-1] + 1, dim=0)

    offset_view = offsets.view(B, 1, 1)
    expanded_offset = offset_view.expand_as(instance_label)
    mask_gt_0 = instance_label > 0

    global_inst = instance_label.clone()
    global_inst[mask_gt_0] += expanded_offset[mask_gt_0]

    max_inst_id = int(global_inst.max().item())
    id_mapper = torch.zeros(max_inst_id + 1, dtype=class_label.dtype, device=device)

    valid_mask = global_inst > 0
    inst_valid = global_inst[valid_mask].long()
    class_valid = class_label[valid_mask].long()
    core_valid = core_mask[valid_mask]

    max_cls_id = int(class_label.max().item()) + 1

    inst_nc = inst_valid[~core_valid]
    class_nc = class_valid[~core_valid]
    if len(inst_nc) > 0:
        comb_nc = inst_nc * max_cls_id + class_nc
        counts_nc = torch.bincount(comb_nc, minlength=(max_inst_id + 1) * max_cls_id).view(max_inst_id + 1, max_cls_id)
        valid_nc_uids = inst_nc.unique()
        id_mapper[valid_nc_uids] = torch.argmax(counts_nc, dim=1)[valid_nc_uids].to(class_label.dtype)

    inst_c = inst_valid[core_valid]
    class_c = class_valid[core_valid]
    if len(inst_c) > 0:
        comb_c = inst_c * max_cls_id + class_c
        counts_c = torch.bincount(comb_c, minlength=(max_inst_id + 1) * max_cls_id).view(max_inst_id + 1, max_cls_id)
        valid_c_uids = inst_c.unique()
        id_mapper[valid_c_uids] = torch.argmax(counts_c, dim=1)[valid_c_uids].to(class_label.dtype)

    # Step 1: Batched expansion
    territory_inst = balanced_dilate_id(global_inst, expand_k)
    territory_type = id_mapper[territory_inst]
    search_zone = (territory_type > 0)

    conf_smoothed = F.avg_pool2d(conf.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)

    # Step 2: Batched anchor selection
    # ``search_zone`` requires ``territory_type > 0``; dilated background can have type 0, so
    # ``n_pred`` (below) may be >0 while ``valid_anchors`` stays empty — no label change, flat global/gene metrics.
    pre_anchor = (prediction == territory_type) & (conf_smoothed > threshold) & search_zone & non_core_zone
    # 3x3 morphological opening on sparse anchors removed almost all pixels; use raw mask — cleanup_islands follows.
    anchors_mask = pre_anchor

    # Step 3: Attribution filtering (skipped when attributions is None)
    stats_before = 0
    stats_after = 0
    has_attr = False

    if attributions is not None:
        has_attr = True
        attributions = attributions.to(device)
        attr_sum_over_gene = attributions.sum(dim=2)
        stats_before = anchors_mask.sum().item()

        gather_index = territory_type.long().unsqueeze(1)
        gather_index = torch.clamp(gather_index, max=attr_sum_over_gene.shape[1] - 1)
        attr_score = attr_sum_over_gene.gather(dim=1, index=gather_index).squeeze(1)

        for b in range(B):
            anchor_scores = attr_score[b][anchors_mask[b]]
            if anchor_scores.numel() > 0:
                q = float(max(0.0, min(1.0, 1.0 - attr_top_ratio)))
                thr = torch.quantile(anchor_scores.float(), q)
                anchors_mask[b] = anchors_mask[b] & (attr_score[b] >= thr)

        stats_after = anchors_mask.sum().item()

    valid_anchors = anchors_mask.bool()

    # Steps 4-5: Batched gap filling
    bridge_mask = balanced_dilate_id(valid_anchors.long(), expand_k) > 0
    fill_zone_mask = bridge_mask & search_zone & non_core_zone

    new_global_inst = global_inst.clone()
    new_global_inst[valid_anchors] = territory_inst[valid_anchors]
    new_global_inst[fill_zone_mask] = territory_inst[fill_zone_mask]

    # Step 6: Connectivity cleanup
    new_global_inst = cleanup_islands(new_global_inst, core_mask)

    # Step 7: Revert offset to original ID space
    mask_gt_0_new = new_global_inst > 0
    new_instance_label = new_global_inst.clone()
    new_instance_label[mask_gt_0_new] -= expanded_offset[mask_gt_0_new]

    # Step 8: Hole filling
    new_instance_label = fill_instance(new_instance_label)

    # Steps 9-10: Map back and restore core
    new_class_label = id_mapper[new_global_inst]

    new_instance_label[core_mask] = instance_label[core_mask]
    new_class_label[core_mask] = class_label[core_mask]

    n_pred = ((prediction == territory_type) & (conf_smoothed > threshold) & non_core_zone).sum().item()
    n_zone = (search_zone & non_core_zone).sum().item()
    is_newly_added = valid_anchors & (global_inst == 0)
    n_upd = is_newly_added.sum().item()
    n_anchors = int(valid_anchors.sum().item())
    frac_cls_changed = float((new_class_label != class_label).float().mean().item())
    frac_inst_changed = float((new_instance_label != instance_label).float().mean().item())

    if batch_id == 0:
        attr_msg = ""
        if has_attr:
            ratio = stats_after / (stats_before + 1e-8)
            attr_msg = f" | attr_Pass: {stats_after}/{stats_before}({ratio:.2%})"

        print(
            f"epoch: {epoch_id} | [Stats] Pred: {n_pred} | Zone: {n_zone} | Upd: {n_upd} | "
            f"Anchors: {n_anchors} | Δcls: {frac_cls_changed:.6f} | Δinst: {frac_inst_changed:.6f}"
            f"{attr_msg}"
        )

    return new_class_label, new_instance_label

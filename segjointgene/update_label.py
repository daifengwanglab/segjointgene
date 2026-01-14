import torch
import torch.nn.functional as F
import cv2

def dilate_id(tensor_id, k):
    return F.max_pool2d(tensor_id.float().unsqueeze(1), kernel_size=2 * k + 1, stride=1, padding=k).squeeze(1).long()

def update_label(class_label, instance_label, core_mask, output, epoch_id, batch_id,
                 expand_k=2, threshold=0.2, attributions=None, target_celltype=None):
    new_instance_label = instance_label.clone()
    new_class_label = class_label.clone()

    probs = F.softmax(output, dim=1)
    conf, prediction = torch.max(probs, dim=1)
    non_core_zone = (~core_mask)
    territory_inst = dilate_id(instance_label, expand_k)
    territory_type = dilate_id(class_label, expand_k)
    search_zone = (territory_type > 0)
    conf_smoothed = F.avg_pool2d(conf.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    anchors_mask = (prediction == territory_type) & (conf_smoothed > threshold) & search_zone & non_core_zone
    attr_msg = ""

    # [New] Constraint: Highest attribution cell type must match territory_type
    if attributions is not None and target_celltype is not None:
        attr_sum_gene = attributions.sum(dim=2).to(output.device)
        max_attr_indices = torch.argmax(attr_sum_gene, dim=1)
        target_celltype_tensor = torch.tensor(target_celltype, device=output.device)
        predicted_attr_class = target_celltype_tensor[max_attr_indices]
        attr_consistent = (predicted_attr_class == territory_type)
        if batch_id == 0:
            n_before_ig = anchors_mask.sum().item()
        anchors_mask = anchors_mask & attr_consistent
        if batch_id == 0:
            n_after_ig = anchors_mask.sum().item()
            ratio = n_after_ig / (n_before_ig + 1e-8)
            attr_msg = f" | attr_Pass: {n_after_ig}/{n_before_ig}({ratio:.2%})"

    if batch_id == 0:
        total = instance_label.numel()
        n_pred = ((prediction == territory_type) & (conf_smoothed > threshold) & non_core_zone).sum().item()
        n_zone = (search_zone & non_core_zone).sum().item()
        n_upd = (anchors_mask).sum().item()
        print(
            f"epoch: {epoch_id} | [Update Stats] Pred: {n_pred}({n_pred / total:.2%}) | Zone: {n_zone}({n_zone / total:.2%}) | Update: {n_upd}({n_upd / total:.2%}){attr_msg}")

    votes = F.conv2d(anchors_mask.float().unsqueeze(1), torch.ones((1, 1, 3, 3)).to(output.device), padding=1).squeeze(1)
    valid_anchors = (anchors_mask) & (votes >= 3)

    bridge_mask = dilate_id(valid_anchors, expand_k) > 0
    fill_zone_mask = bridge_mask & search_zone & non_core_zone

    new_instance_label[valid_anchors] = territory_inst[valid_anchors]
    new_class_label[valid_anchors] = territory_type[valid_anchors]
    new_instance_label[fill_zone_mask] = territory_inst[fill_zone_mask]
    new_class_label[fill_zone_mask] = territory_type[fill_zone_mask]

    return new_class_label, new_instance_label
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import random
matplotlib.use('Agg')
from utils import *

def update_label(class_label, instance_label, core_mask, output, epoch_id, batch_id,
                 expand_k=2, threshold=0.2, attributions=None, target_cells=None):
    new_instance_label = instance_label.clone()
    new_class_label = class_label.clone()
    probs = F.softmax(output, dim=1)
    conf, prediction = torch.max(probs, dim=1)
    non_core_zone = (~core_mask)

    def dilate_id(tensor_id, k):
        return F.max_pool2d(
            tensor_id.float().unsqueeze(1),
            kernel_size=2 * k + 1,
            stride=1,
            padding=k
        ).squeeze(1).long()

    territory_inst = dilate_id(instance_label, expand_k)
    territory_type = dilate_id(class_label, expand_k)
    search_zone = (territory_type > 0)
    conf_smoothed = F.avg_pool2d(conf.unsqueeze(1), 3, 1, 1).squeeze(1)
    anchors_mask = (
        (prediction == territory_type)
        & (conf_smoothed > threshold)
        & search_zone
        & non_core_zone
    )

    if attributions is not None and target_cells is not None:
        attr_sum_genes = attributions.sum(dim=2).to(output.device)
        max_attr_indices = torch.argmax(attr_sum_genes, dim=1)
        target_cells_tensor = torch.tensor(target_cells, device=output.device)
        predicted_attr_class = target_cells_tensor[max_attr_indices]
        anchors_mask = anchors_mask & (predicted_attr_class == territory_type)

    votes = F.conv2d(
        anchors_mask.float().unsqueeze(1),
        torch.ones((1, 1, 3, 3), device=output.device),
        padding=1
    ).squeeze(1)

    valid_anchors = anchors_mask & (votes >= 3)
    bridge_mask = dilate_id(valid_anchors, expand_k) > 0
    fill_zone_mask = bridge_mask & search_zone & non_core_zone

    new_instance_label[valid_anchors] = territory_inst[valid_anchors]
    new_class_label[valid_anchors] = territory_type[valid_anchors]
    new_instance_label[fill_zone_mask] = territory_inst[fill_zone_mask]
    new_class_label[fill_zone_mask] = territory_type[fill_zone_mask]

    return new_class_label, new_instance_label

def get_target_selection(train_loader, args):
    total_gene_sum = None
    total_class_counts = None

    with torch.no_grad():
        for batch_data in train_loader:
            image, label = batch_data[0], batch_data[1]
            current_sum = image.sum(dim=(0, 2, 3)).cpu()
            if total_gene_sum is None:
                total_gene_sum = current_sum
            else:
                min_c = min(total_gene_sum.shape[0], current_sum.shape[0])
                total_gene_sum[:min_c] += current_sum[:min_c]

            label_flat = label.flatten().cpu().long()
            num_classes = label_flat.max().item() + 1
            current_counts = torch.bincount(label_flat, minlength=num_classes)

            if total_class_counts is None:
                total_class_counts = current_counts
            else:
                if current_counts.shape[0] > total_class_counts.shape[0]:
                    total_class_counts = torch.cat([
                        total_class_counts,
                        torch.zeros(
                            current_counts.shape[0] - total_class_counts.shape[0],
                            dtype=total_class_counts.dtype
                        )
                    ])
                elif total_class_counts.shape[0] > current_counts.shape[0]:
                    current_counts = torch.cat([
                        current_counts,
                        torch.zeros(
                            total_class_counts.shape[0] - current_counts.shape[0],
                            dtype=current_counts.dtype
                        )
                    ])
                total_class_counts += current_counts

    sorted_genes = torch.argsort(total_gene_sum, descending=True).tolist()
    sorted_classes = torch.argsort(total_class_counts, descending=True).tolist()
    filtered_classes = [c for c in sorted_classes if c != 0]

    try:
        grid_str = str(getattr(args, 'attr_grid', '1x1')).replace('*', 'x')
        grid_n = int(grid_str.split('x')[0]) if 'x' in grid_str else int(grid_str)
    except:
        grid_n = 1

    real_gene_n = min(grid_n, len(sorted_genes))
    real_class_n = min(grid_n, len(filtered_classes))

    if real_class_n == 0 and len(sorted_classes) > 0:
        filtered_classes = [0]
        real_class_n = 1

    target_genes = sorted_genes[:real_gene_n]
    target_cells = filtered_classes[:real_class_n]
    return target_genes, target_cells

def compute_cid_matrix(
    net, input_img, target_genes, target_cells,
    n_steps=20, lr=0.1, lambda_param=0.01, beta=1.2, chunk_size=None
):
    was_training = net.training
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    clean_input = input_img.detach().clone()
    B, C, H, W = clean_input.shape

    with torch.no_grad():
        clean_output = net(clean_input).detach()

    n_cells = len(target_cells)
    n_genes = len(target_genes)
    batch_attr = torch.zeros((B, n_cells, n_genes, H, W), device='cpu')

    if chunk_size is None:
        chunk_size = n_cells

    for i in range(0, n_cells, chunk_size):
        try:
            current_indices = range(i, min(i + chunk_size, n_cells))
            current_cell_types = [target_cells[idx] for idx in current_indices]
            current_chunk_len = len(current_cell_types)

            input_expanded = clean_input.repeat(current_chunk_len, 1, 1, 1)

            target_maps = []
            for c_idx in current_cell_types:
                target_maps.append(clean_output[:, c_idx, :, :])
            target_maps = torch.cat(target_maps, dim=0)

            log_sigma = torch.full_like(input_expanded, -5.0, requires_grad=True)
            optimizer = torch.optim.Adam([log_sigma], lr=lr)

            N_total = input_expanded.shape[0]
            initial_means = None
            finished_mask = torch.zeros(N_total, dtype=torch.bool, device=clean_input.device)

            for step in range(n_steps):
                optimizer.zero_grad()
                sigma = torch.exp(log_sigma)
                epsilon = torch.randn_like(input_expanded)
                noisy_input = input_expanded + sigma * epsilon
                noisy_output = net(noisy_input)

                cell_indices = torch.tensor(
                    current_cell_types,
                    device=noisy_output.device
                ).repeat_interleave(B)

                current_maps = noisy_output[
                    torch.arange(noisy_output.size(0)),
                    cell_indices,
                    :, :
                ]

                loss_mse = F.mse_loss(
                    current_maps, target_maps, reduction='none'
                ).mean(dim=(1, 2))
                loss_entropy = -log_sigma.mean(dim=(1, 2, 3))
                loss_per_sample = loss_mse + lambda_param * loss_entropy
                loss_per_sample.sum().backward()

                with torch.no_grad():
                    current_means = torch.exp(log_sigma).mean(dim=(1, 2, 3))
                    if step == 0:
                        initial_means = current_means
                    else:
                        ratios = current_means / initial_means
                        newly_finished = (ratios > beta) & (~finished_mask)
                        if newly_finished.any():
                            finished_mask |= newly_finished

                if finished_mask.any():
                    log_sigma.grad[
                        finished_mask.view(-1, 1, 1, 1).expand_as(log_sigma.grad)
                    ] = 0.0

                optimizer.step()
                if finished_mask.all():
                    break

            sigma_data = log_sigma.detach()
            final_attr = -0.5 * math.pi * math.e * sigma_data
            final_attr = final_attr.view(current_chunk_len, B, C, H, W)

            for local_i, global_i in enumerate(current_indices):
                batch_attr[:, global_i, :, :, :] = \
                    final_attr[local_i][:, target_genes, :, :].cpu()

            del log_sigma, optimizer, final_attr, target_maps, input_expanded, noisy_output, sigma_data
            torch.cuda.empty_cache()

        except RuntimeError:
            torch.cuda.empty_cache()

    for p in net.parameters():
        p.requires_grad = True
    if was_training:
        net.train()

    return net, batch_attr


def step_SegJointGene_CID(root_path, args):
    path_dict, step_suffix = step_set_path(root_path, args)
    setup_seed(args.random_seed)

    train_set, test_set, train_loader, test_loader = step_get_datasets_loader(path_dict, args)

    target_genes, target_cells = get_target_selection(train_loader, args)

    net = get_net(net_name='unet', args=args)
    net = step_set_seed(path_dict, net)
    optimizer = step_get_optimizer(net, args)
    criterion = nn.CrossEntropyLoss().cuda()
    net, optimizer, _, start_epoch = step_load_ckpt(
        path_dict, net, optimizer, None, args, if_load=args.if_load_ckpt
    )

    if args.if_load_ckpt:
        step_load_label_cache(path_dict, start_epoch, train_set, test_set)

    for epoch_id in range(start_epoch, args.net_epoch + 1):
        run_cid = (epoch_id >= args.attr_epoch)

        net.train()
        for batch_id, batch_data in enumerate(train_loader):
            image, label, instance_label, spots, dapi, idx, rows, cols, fixed_label, fixed_inst = batch_data
            image = image.cuda()
            label = label.cuda().long()
            instance_label = instance_label.cuda()
            fixed_inst = fixed_inst.cuda().long()

            output = net(image)
            loss = criterion(output, label)

            if epoch_id != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if run_cid:
                net, attrs = compute_cid_matrix(
                    net, image, target_genes, target_cells,
                    n_steps=args.CID_n_steps,
                    lr=args.CID_lr,
                    lambda_param=args.CID_lambda_param,
                    beta=args.CID_beta,
                    chunk_size=args.CID_chunk_size
                )
            else:
                attrs = None

            if epoch_id == 0:
                new_label = label.clone()
                new_inst = instance_label.clone()
            else:
                new_label, new_inst = update_label(
                    label,
                    instance_label,
                    (fixed_inst > 0),
                    output,
                    epoch_id,
                    batch_id,
                    attributions=attrs,
                    target_cells=target_cells
                )

            train_set.update_label_cache(idx, new_label, new_inst)

        net.eval()
        for batch_id, batch_data in enumerate(test_loader):
            image, label, instance_label, spots, dapi, idx, rows, cols, fixed_label, fixed_inst = batch_data
            image = image.cuda()
            label = label.cuda().long()
            instance_label = instance_label.cuda()
            fixed_inst = fixed_inst.cuda().long()

            with torch.no_grad():
                output = net(image)

            if run_cid:
                net, attrs = compute_cid_matrix(
                    net, image, target_genes, target_cells,
                    n_steps=args.CID_n_steps,
                    lr=args.CID_lr,
                    lambda_param=args.CID_lambda_param,
                    beta=args.CID_beta,
                    chunk_size=args.CID_chunk_size
                )
            else:
                attrs = None

            if epoch_id == 0:
                new_label = label.clone()
                new_inst = instance_label.clone()
            else:
                new_label, new_inst = update_label(
                    label,
                    instance_label,
                    (fixed_inst > 0),
                    output,
                    epoch_id,
                    batch_id,
                    attributions=attrs,
                    target_cells=target_cells
                )

            test_set.update_label_cache(idx, new_label, new_inst)

        if epoch_id % 10 == 0:
            step_save_label_cache(path_dict, epoch_id, train_set, test_set)
            step_save_ckpt(path_dict, epoch_id, net, optimizer, None, args)

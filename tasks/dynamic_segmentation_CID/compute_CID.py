"""
CID (Cell-type Integrated Gradients) attribution.

Gene-stacking with Monte Carlo noise sampling at grid resolution.
Pure FP16 + channels_last for VRAM efficiency on Ampere+ GPUs.
"""

import collections
import math
import time

import torch

MAX_CONCURRENT_SAMPLES = 600


def compute_CID(
    net,
    input_img,
    target_gene,
    target_celltype,
    label=None,
    instance_label=None,
    n_steps=20,
    lr=0.1,
    lambda_param=0.01,
    beta=1.2,
    noise_num=8,
    gene_chunk_size=100,
    grid_size=8,
    return_spatial=False,
):
    """
    CID with gene-stacking (parallel isolation).

    Unfinished pairs after *n_steps* default to log_sigma = -2.0.
    Returns ``(net, batch_attr_cpu, time_stats_dict)``.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def sync_t():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    time_stats = collections.defaultdict(float)
    t_total_start = sync_t()
    t_curr = sync_t()

    was_training = net.training
    net.eval()

    net = net.half().to(memory_format=torch.channels_last)
    for p in net.parameters():
        p.requires_grad = False

    time_stats["01_Model_Prep"] += sync_t() - t_curr

    try:
        t_curr = sync_t()
        clean_input = input_img.detach().half().contiguous(memory_format=torch.channels_last)
        device = clean_input.device
        B_real, C, H, W = clean_input.shape
        assert H % grid_size == 0 and W % grid_size == 0

        Hg = H // grid_size
        Wg = W // grid_size

        with torch.no_grad():
            _out = net(clean_input)
            clean_output = (_out[0] if isinstance(_out, (tuple, list)) else _out).detach()

        time_stats["02_Clean_Forward"] += sync_t() - t_curr
        t_curr = sync_t()

        cell_indices = torch.as_tensor(target_celltype, device=device)
        n_celltype = len(target_celltype)
        n_gene = len(target_gene)
        target_gene_t = torch.tensor(target_gene, device=device)

        if return_spatial:
            batch_attr = torch.zeros((B_real, n_celltype, n_gene, H, W), dtype=torch.float16, device=device)
        else:
            batch_attr = torch.zeros((B_real, n_celltype, n_gene), dtype=torch.float32, device=device)

        tasks = []
        for b in range(B_real):
            for c in range(n_celltype):
                for g in range(n_gene):
                    tasks.append((b, c, g))

        total_tasks = len(tasks)
        print("total tasks:", total_tasks)

        time_stats["03_Task_Setup"] += sync_t() - t_curr

        for t_start in range(0, total_tasks, MAX_CONCURRENT_SAMPLES):
            try:
                t_curr = sync_t()
                t_end = min(t_start + MAX_CONCURRENT_SAMPLES, total_tasks)
                curr_batch_tasks = tasks[t_start:t_end]
                curr_batch_len = len(curr_batch_tasks)

                batch_sample_indices = [t[0] for t in curr_batch_tasks]
                batch_cell_indices = [t[1] for t in curr_batch_tasks]
                batch_gene_indices = [t[2] for t in curr_batch_tasks]

                batch_sample_indices_t = torch.tensor(batch_sample_indices, device=device)
                batch_cell_indices_t = torch.tensor(batch_cell_indices, device=device)
                batch_gene_indices_t = torch.tensor(batch_gene_indices, device=device)

                sub_input = clean_input[batch_sample_indices].clone()

                with torch.no_grad():
                    real_cell_ids = cell_indices[batch_cell_indices_t]
                    target_subset = clean_output[batch_sample_indices_t, real_cell_ids]

                time_stats["04_Chunk_Initialization"] += sync_t() - t_curr

                log_sigma = torch.full(
                    (curr_batch_len, 1, Hg, Wg),
                    -3.0,
                    device=device,
                    requires_grad=True,
                )

                base_epsilon = torch.randn((curr_batch_len, noise_num, Hg, Wg), device=device)
                optimizer = torch.optim.Adam([log_sigma], lr=lr)

                finished_mask = torch.zeros((curr_batch_len,), dtype=torch.bool, device=device)
                base_mse = torch.full((curr_batch_len,), float("nan"), device=device)

                dynamic_lambda = torch.full((curr_batch_len,), lambda_param, device=device, dtype=torch.float32)
                dynamic_lambda.clamp_(0.001, 1000.0)
                prev_ratios = torch.ones((curr_batch_len,), device=device, dtype=torch.float32)

                for step in range(n_steps):
                    t_step = sync_t()
                    optimizer.zero_grad(set_to_none=True)

                    active_mask = ~finished_mask
                    active_idx = active_mask.nonzero(as_tuple=True)[0]
                    N_active = len(active_idx)

                    if N_active == 0:
                        break
                    time_stats["05_Step_OptimZero_ActiveCheck"] += sync_t() - t_step

                    t_step = sync_t()
                    sigma = torch.exp(log_sigma[active_idx])
                    curr_epsilon = base_epsilon[active_idx]

                    noise_grid = sigma * curr_epsilon
                    noise_to_add = (
                        noise_grid.repeat_interleave(grid_size, dim=2).repeat_interleave(grid_size, dim=3)
                    )

                    noisy_input = sub_input[active_idx].unsqueeze(1).repeat(1, noise_num, 1, 1, 1)

                    current_g_indices = batch_gene_indices_t[active_idx]
                    range_idx = torch.arange(N_active, device=device)
                    real_channel_ids = target_gene_t[current_g_indices]

                    noisy_input[
                        range_idx[:, None],
                        torch.arange(noise_num, device=device)[None, :],
                        real_channel_ids[:, None],
                    ] += noise_to_add.half()

                    noisy_input_flat = noisy_input.view(-1, C, H, W)
                    noisy_input_flat = noisy_input_flat.contiguous(memory_format=torch.channels_last)

                    time_stats["06_Step_Noise_Injection"] += sync_t() - t_step

                    t_step = sync_t()
                    _noisy_out = net(noisy_input_flat)
                    noisy_output_flat = _noisy_out[0] if isinstance(_noisy_out, (tuple, list)) else _noisy_out

                    active_cell_ids = real_cell_ids[active_idx]
                    active_cell_ids_flat = active_cell_ids.repeat_interleave(noise_num)

                    current_subset_flat = noisy_output_flat[
                        torch.arange(N_active * noise_num, device=device), active_cell_ids_flat
                    ]

                    time_stats["07_Step_Noisy_Forward"] += sync_t() - t_step

                    t_step = sync_t()
                    target_f = target_subset[active_idx].float().unsqueeze(1)

                    if current_subset_flat.ndim == 3:
                        _, Hout, Wout = current_subset_flat.shape
                        current_subset = current_subset_flat.view(N_active, noise_num, Hout, Wout)
                        mse_per_cell_all = (current_subset.float() - target_f).pow(2).mean(dim=(2, 3))
                    else:
                        current_subset = current_subset_flat.view(N_active, noise_num)
                        mse_per_cell_all = (current_subset.float() - target_f).pow(2)

                    mse_per_cell = mse_per_cell_all.mean(dim=1)

                    if step == 0:
                        base_mse[active_idx] = mse_per_cell.detach()
                    time_stats["08_Step_MSE_Calculation"] += sync_t() - t_step

                    t_step = sync_t()
                    with torch.no_grad():
                        ratios = mse_per_cell / base_mse[active_idx].clamp_min(1e-12)

                        if step > 0:
                            delta = ratios - prev_ratios[active_idx]
                            error = 0.01 - delta
                            scale_factor = torch.exp(50.0 * error)
                            scale_factor = scale_factor.clamp(0.5, 2.0)
                            dynamic_lambda[active_idx] *= scale_factor
                            dynamic_lambda.clamp_(0.001, 1000.0)

                        prev_ratios[active_idx] = ratios.clone()

                        finished_idx = ratios > beta
                        newly_finished = finished_idx & (~finished_mask[active_idx])

                    time_stats["09_Step_Ratio_Lambda_Update"] += sync_t() - t_step

                    t_step = sync_t()
                    loss_mse = mse_per_cell
                    loss_entropy = -log_sigma[active_idx].mean(dim=(1, 2, 3))
                    loss = (loss_mse + dynamic_lambda[active_idx] * loss_entropy).sum()

                    loss.backward()

                    del noisy_input_flat, noisy_output_flat, noisy_input, noise_to_add, noise_grid
                    del current_subset_flat, current_subset, target_f, mse_per_cell_all
                    time_stats["10_Step_Loss_Backward"] += sync_t() - t_step

                    t_step = sync_t()
                    optimizer.step()
                    time_stats["11_Step_Optimizer_Step"] += sync_t() - t_step

                    t_step = sync_t()
                    with torch.no_grad():
                        if newly_finished.any():
                            attr_snapshot = (
                                log_sigma[active_idx]
                                .repeat_interleave(grid_size, dim=2)
                                .repeat_interleave(grid_size, dim=3)
                            )
                            attr_snapshot *= -0.5 * math.pi * math.e

                            ai = newly_finished.nonzero(as_tuple=True)[0]
                            batch_idx_in_chunk = active_idx[ai]

                            rs_tensor = batch_sample_indices_t[batch_idx_in_chunk]
                            rc_tensor = batch_cell_indices_t[batch_idx_in_chunk]
                            rg_tensor = batch_gene_indices_t[batch_idx_in_chunk]

                            if return_spatial:
                                batch_attr[rs_tensor, rc_tensor, rg_tensor] = attr_snapshot[ai].squeeze(1).half()
                            else:
                                attr_mean = attr_snapshot.mean(dim=(2, 3)).squeeze(1)
                                batch_attr[rs_tensor, rc_tensor, rg_tensor] = attr_mean[ai]

                            finished_mask[active_idx] |= newly_finished

                    del loss, loss_mse, loss_entropy, mse_per_cell, sigma
                    time_stats["12_Step_Finished_Logic"] += sync_t() - t_step

                t_curr = sync_t()
                with torch.no_grad():
                    still = ~finished_mask
                    if still.any():
                        attr_final = (
                            log_sigma.repeat_interleave(grid_size, dim=2).repeat_interleave(grid_size, dim=3)
                        )
                        attr_final *= -0.5 * math.pi * math.e

                        bi = still.nonzero(as_tuple=True)[0]
                        rs_tensor = batch_sample_indices_t[bi]
                        rc_tensor = batch_cell_indices_t[bi]
                        rg_tensor = batch_gene_indices_t[bi]

                        if return_spatial:
                            batch_attr[rs_tensor, rc_tensor, rg_tensor] = attr_final[bi].squeeze(1).half()
                        else:
                            attr_mean = attr_final.mean(dim=(2, 3)).squeeze(1)
                            batch_attr[rs_tensor, rc_tensor, rg_tensor] = attr_mean[bi]

                del log_sigma, optimizer
                time_stats["13_Chunk_Unfinished_Logic"] += sync_t() - t_curr

            except RuntimeError as e:
                print(f"[CID Error] Task chunk {t_start}: {e}")

    finally:
        t_curr = sync_t()
        net.float()
        for p in net.parameters():
            p.requires_grad = True
        if was_training:
            net.train()
        time_stats["14_Cleanup_Restore"] += sync_t() - t_curr

    time_stats["00_Total"] = max(0.0, sync_t() - t_total_start)
    time_stats_out = {str(k): float(v) for k, v in sorted(time_stats.items(), key=lambda kv: kv[0])}
    return net, batch_attr.cpu(), time_stats_out

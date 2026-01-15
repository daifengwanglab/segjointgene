import cv2
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

MAX_CONCURRENT_SAMPLES = 10

def compute_CID(net, input_img, target_gene, target_celltype,
                n_steps=20, lr=0.1, lambda_param=0.01, beta=1.2,
                cell_chunk_size=1, gene_chunk_size=100,
                return_spatial=False):

    was_training = net.training
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    clean_input = input_img.detach()
    B, C, H, W = clean_input.shape

    with torch.no_grad():
        clean_output = net(clean_input).detach()

    n_celltype = len(target_celltype)
    n_gene = len(target_gene)

    if return_spatial:
        batch_attr = torch.zeros((B, n_celltype, n_gene, H, W),
                                 dtype=torch.float16, device='cpu')
    else:
        batch_attr = torch.zeros((B, n_celltype, n_gene),
                                 dtype=torch.float32, device='cpu')

    scaler = GradScaler()

    for i in range(0, n_celltype, cell_chunk_size):
        current_cell_indices = list(range(i, min(i + cell_chunk_size, n_celltype)))
        current_cell_types = [target_celltype[idx] for idx in current_cell_indices]
        c_chunk_len = len(current_cell_types)

        sub_batch_size = max(1, MAX_CONCURRENT_SAMPLES // c_chunk_len)

        for b_start in range(0, B, sub_batch_size):
            try:
                b_end = min(b_start + sub_batch_size, B)
                curr_batch_len = b_end - b_start
                N_total = curr_batch_len * c_chunk_len

                sub_output = clean_output[b_start:b_end]
                target_subset = sub_output[:, current_cell_types]
                target_maps = target_subset.permute(1, 0, 2, 3).reshape(N_total, H, W)

                sub_input = clean_input[b_start:b_end]
                input_expanded_base = (
                    sub_input.unsqueeze(0)
                    .expand(c_chunk_len, -1, -1, -1, -1)
                    .reshape(N_total, C, H, W)
                )

                gather_indices = torch.tensor(
                    current_cell_types,
                    device=clean_input.device
                ).repeat_interleave(curr_batch_len)

                for g_start in range(0, n_gene, gene_chunk_size):
                    g_end = min(g_start + gene_chunk_size, n_gene)
                    curr_gene_indices_local = list(range(g_start, g_end))
                    curr_input_channels = [target_gene[idx] for idx in curr_gene_indices_local]
                    n_curr_gene = len(curr_input_channels)

                    log_sigma = torch.full(
                        (N_total, n_curr_gene, H, W),
                        -5.0,
                        device=clean_input.device,
                        requires_grad=True
                    )

                    optimizer = torch.optim.Adam([log_sigma], lr=lr)

                    finished_mask = torch.zeros(
                        N_total, dtype=torch.bool, device=clean_input.device
                    )

                    initial_means_full = torch.empty(
                        N_total, device=clean_input.device, dtype=torch.float32
                    )
                    initial_means_full.fill_(float('nan'))

                    for step in range(n_steps):
                        optimizer.zero_grad()
                        active_idx = (~finished_mask).nonzero(as_tuple=True)[0]
                        if active_idx.numel() == 0:
                            break

                        with autocast():
                            sigma = torch.exp(log_sigma[active_idx])
                            epsilon = torch.randn_like(sigma)

                            noisy_input = input_expanded_base[active_idx].clone(
                                memory_format=torch.preserve_format
                            )
                            noisy_input[:, curr_input_channels] += sigma * epsilon

                            noisy_output = net(noisy_input)
                            current_maps = noisy_output[
                                torch.arange(len(active_idx), device=noisy_output.device),
                                gather_indices[active_idx]
                            ]

                        current_maps_f = current_maps.float()
                        target_maps_f = target_maps[active_idx].float()

                        loss_mse = F.mse_loss(
                            current_maps_f,
                            target_maps_f,
                            reduction='none'
                        ).mean(dim=(1, 2))

                        loss_entropy = -log_sigma[active_idx].float().mean(dim=(1, 2, 3))
                        loss_per_sample = loss_mse + lambda_param * loss_entropy
                        total_loss = loss_per_sample.sum()

                        scaler.scale(total_loss).backward()

                        with torch.no_grad():
                            current_means = sigma.mean(dim=(1, 2, 3)).float()
                            if step == 0:
                                initial_means_full[active_idx] = current_means
                            else:
                                base = initial_means_full[active_idx]
                                ratios = current_means / base
                                newly_finished = ratios > beta
                                if newly_finished.any():
                                    finished_mask[active_idx[newly_finished]] = True

                        scaler.step(optimizer)
                        scaler.update()

                    with torch.no_grad():
                        log_sigma_view = log_sigma.view(
                            c_chunk_len, curr_batch_len, n_curr_gene, H, W
                        )

                        const_factor = -0.5 * math.pi * math.e
                        final_attr = log_sigma_view * const_factor

                        if return_spatial:
                            batch_attr[
                                b_start:b_end,
                                i:i + c_chunk_len,
                                g_start:g_end
                            ] = final_attr.permute(1, 0, 2, 3, 4).cpu().half()
                        else:
                            attr_mean = final_attr.mean(dim=(3, 4))
                            batch_attr[
                                b_start:b_end,
                                i:i + c_chunk_len,
                                g_start:g_end
                            ] = attr_mean.permute(1, 0, 2).cpu()

                    del log_sigma, optimizer, noisy_output, noisy_input, loss_mse
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"\n[CID Error] CellChunk {i} Batch {b_start}: {e}")
                torch.cuda.empty_cache()

    for p in net.parameters():
        p.requires_grad = True
    if was_training:
        net.train()

    return net, batch_attr
import os
import torch
import torch.nn.functional as F
import cv2

class GlobalStitchingEvaluator:
    def __init__(self, test_set, train_set, patch_size, pixel_distance, target_gene, target_celltypes, target_gene_names, target_celltype_names, if_attr):
        self.patch_size = patch_size
        self.d = pixel_distance
        self.kernel_size = 2 * self.d + 1
        self.epsilon = 1e-6

        self.target_gene = target_gene
        self.target_celltypes = target_celltypes
        self.target_gene_names = target_gene_names
        self.target_celltype_names = target_celltype_names
        self.n_gene = len(target_gene)
        self.n_celltype = len(target_celltypes)
        self.if_attr = if_attr

        max_row = 0
        max_col = 0
        all_files = test_set.file_list + train_set.file_list

        for path in all_files:
            basename = os.path.basename(path)
            if not basename.startswith("p_"):
                continue
            try:
                parts = basename.split('.')[0].split('_')
                r, c = int(parts[1]), int(parts[2])
                if r > max_row: max_row = r
                if c > max_col: max_col = c
            except:
                continue

        self.rows_count = max_row + 1
        self.cols_count = max_col + 1
        self.H_global = self.rows_count * patch_size
        self.W_global = self.cols_count * patch_size

        print(f"\n[Stitcher Init] Global Size: {self.W_global}x{self.H_global}")
        print(f"[Stitcher Init] Grid Config: {self.n_celltype} CellTypes x {self.n_gene} Genes")

        self.global_mask = torch.zeros((self.H_global, self.W_global), dtype=torch.long)
        self.global_spots = torch.zeros((self.H_global, self.W_global), dtype=torch.float32)
        self.occurrence_map = torch.zeros((self.rows_count, self.cols_count), dtype=torch.int32)
        # (N_cell, N_gene, H, W) - float32 required for small gradients
        if self.if_attr:
            self.global_attr = torch.zeros((self.n_celltype, self.n_gene, self.H_global, self.W_global), dtype=torch.float32)

    def update(self, labels, spots, rows, cols, attributions=None):
        if self.if_attr:
            assert attributions != None
        else:
            assert attributions == None

        labels = labels.detach().cpu()
        spots = spots.detach().cpu()
        rows = rows.detach().cpu().numpy()
        cols = cols.detach().cpu().numpy()

        if attributions is not None:
            attributions = attributions.detach().cpu()

        B = labels.shape[0]
        for i in range(B):
            r, c = rows[i], cols[i]
            self.occurrence_map[r, c] += 1

            y_start, x_start = r * self.patch_size, c * self.patch_size
            y_end, x_end = y_start + self.patch_size, x_start + self.patch_size

            self.global_mask[y_start:y_end, x_start:x_end] = labels[i]
            self.global_spots[y_start:y_end, x_start:x_end] = spots[i]

            if attributions is not None:
                self.global_attr[:, :, y_start:y_end, x_start:x_end] = attributions[i]

    @torch.no_grad()
    def compute_score(self):
        g_mask_gpu = self.global_mask.cuda()
        g_spots_gpu = self.global_spots.cuda()

        mask_in = (g_mask_gpu > 0)
        mask_in_float = mask_in.float().unsqueeze(0).unsqueeze(0)

        mask_dilated = F.max_pool2d(
            mask_in_float, kernel_size=self.kernel_size, stride=1, padding=self.d
        ).squeeze() > 0
        mask_margin = mask_dilated & (~mask_in)

        total_gin = (g_spots_gpu * mask_in).sum().item()
        total_gout = (g_spots_gpu * mask_margin).sum().item()

        del g_mask_gpu, g_spots_gpu, mask_in, mask_dilated, mask_margin
        torch.cuda.empty_cache()

        if total_gout == 0: return 0.0
        score = total_gin / (total_gout + self.epsilon)
        print(f"Final Score: {score:.4f}")
        return score
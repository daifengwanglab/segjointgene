"""Simulation dataset: NPZ patch loader with Min-Max normalization for synthetic spatial data."""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset


class ImagePatchDataset(Dataset):
    def __init__(self, npz_dir):
        self.file_list = glob.glob(os.path.join(npz_dir, "p_*.npz"))
        self.file_list.sort()
        print(npz_dir)
        print(f"[Dataset] Loaded dir {os.path.basename(npz_dir)}, files: {len(self.file_list)}")

        self.dynamic_instance_labels = {}
        self.dynamic_class_labels = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        basename = os.path.basename(path)
        name_parts = basename.split('.')[0].split('_')

        group_row = int(name_parts[1])
        group_col = int(name_parts[2])
        row = int(name_parts[3])
        col = int(name_parts[4])

        with np.load(path, allow_pickle=True) as data:
            image = torch.from_numpy(data['image']).float()

            img_max = image.max()
            if img_max > 0:
                image = image / img_max

            spots = torch.from_numpy(data['spots']).float()
            dapi = torch.from_numpy(data['dapi']).float()

            raw_label = torch.from_numpy(data['label']).long()
            raw_instance = torch.from_numpy(data['instance']).long()

            fixed_class_label = raw_label
            fixed_instance_label = raw_instance

            if 'ground_truth' in data and data['ground_truth'].shape != ():
                ground_truth = torch.from_numpy(data['ground_truth']).long()
            else:
                ground_truth = torch.zeros_like(raw_label)

            if 'ground_truth_instance' in data.files and data['ground_truth_instance'].shape != ():
                ground_truth_instance = torch.from_numpy(
                    np.asarray(data['ground_truth_instance'])
                ).long()
            else:
                ground_truth_instance = torch.zeros_like(raw_instance)

            if idx in self.dynamic_class_labels:
                current_class_label = self.dynamic_class_labels[idx]
            else:
                current_class_label = fixed_class_label.clone()

            if idx in self.dynamic_instance_labels:
                current_instance_label = self.dynamic_instance_labels[idx]
            else:
                current_instance_label = fixed_instance_label.clone()

        return (
            image,
            current_class_label,
            current_instance_label,
            spots,
            dapi,
            idx,
            group_row,
            group_col,
            row,
            col,
            fixed_class_label,
            fixed_instance_label,
            ground_truth,
            ground_truth_instance,
        )

    def update_label_cache(self, indices, new_class_labels, new_instance_labels):
        for i, idx_tensor in enumerate(indices):
            idx = idx_tensor.item()
            self.dynamic_class_labels[idx] = new_class_labels[i].detach().cpu().clone()
            self.dynamic_instance_labels[idx] = new_instance_labels[i].detach().cpu().clone()

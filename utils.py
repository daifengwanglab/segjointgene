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

from torch.utils.data import Dataset
from joblib import Parallel, delayed
from tqdm import tqdm

def make_dir(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def _worker_save_sample(idx, image, label, spots, save_dir):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()
    if isinstance(spots, torch.Tensor):
        spots = spots.detach().cpu().numpy()

    file_name = f"sample_{idx}.npz"
    save_path = os.path.join(save_dir, file_name)
    np.savez_compressed(save_path, image=image, label=label, spots=spots)


def step_save_dataset_state(dataset, path_dict, epoch_id, prefix='train', n_jobs=8):
    state_root = os.path.join(path_dict['experiment_dataset_net_path'], 'dataset_states')
    save_dir = os.path.join(state_root, f'epoch_{epoch_id}_{prefix}')
    os.makedirs(save_dir, exist_ok=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False
    )

    tasks = []
    for batch_data in loader:
        input_tensor, label_tensor, spots_tensor, index_tensor = batch_data
        idx = index_tensor.item()
        img = input_tensor.squeeze(0)
        lb = label_tensor.squeeze(0)
        sp = spots_tensor.squeeze(0)
        tasks.append(delayed(_worker_save_sample)(idx, img, lb, sp, save_dir))

    Parallel(n_jobs=n_jobs, backend='threading')(
        tqdm(tasks, total=len(tasks), desc=f"Saving {prefix} set")
    )


def step_get_suffix(args):
    if args.step_name.startswith('net'):
        return args.step_name
    elif args.step_name.startswith('SegjointGene'):
        return 'SegjointGene'
    elif args.step_name.startswith('evaluation') or args.step_name.startswith('visualize'):
        return 'net'
    else:
        raise NameError('Can not recognize the name of step')


def step_set_path(root_path, args):
    experiment_path = os.path.join(root_path, 'experiment')
    data_path = os.path.join(root_path, 'data')

    make_dir(experiment_path)
    make_dir(data_path)

    experiment_dataset_path = os.path.join(experiment_path, args.datasets_name)
    data_dataset_path = os.path.join(data_path, args.datasets_name)

    make_dir(experiment_dataset_path)
    make_dir(data_dataset_path)

    step_suffix = step_get_suffix(args)

    experiment_dataset_net_path = os.path.join(experiment_dataset_path, args.net_name)
    make_dir(experiment_dataset_net_path)

    net_path = os.path.join(experiment_dataset_net_path, step_suffix)
    make_dir(net_path)

    net_sub_path = os.path.join(net_path, args.net_sub_suffix)
    make_dir(net_sub_path)

    path_dict = {
        'root_path': root_path,
        'experiment_path': experiment_path,
        'data_path': data_path,
        'experiment_dataset_path': experiment_dataset_path,
        'data_dataset_path': data_dataset_path,
        'experiment_dataset_net_path': experiment_dataset_net_path,
        'net_path': net_path,
        'net_sub_path': net_sub_path,
    }
    return path_dict, step_suffix


def step_get_datasets_loader(path_dict, args):
    train_set = get_datasets(args.datasets_name, data_path=path_dict['data_dataset_path'], train=True, args=args)
    test_set = get_datasets(args.datasets_name, data_path=path_dict['data_dataset_path'], train=False, args=args)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.net_batch_size, shuffle=True, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.net_batch_size, shuffle=False, num_workers=args.num_workers
    )

    assert len(train_set) != 0
    assert len(test_set) != 0
    return train_set, test_set, train_loader, test_loader


def step_set_seed(path_dict, net):
    net_seed_path = os.path.join(path_dict['net_sub_path'], 'net_seed.bin')
    if os.path.exists(net_seed_path):
        net.load_state_dict(torch.load(net_seed_path))
    else:
        torch.save(net.state_dict(), net_seed_path)
    return net


def step_get_optimizer(net, args):
    if args.net_optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            [{"params": net.parameters(), 'lr': args.net_lr, 'initial_lr': args.net_lr}],
            weight_decay=args.net_weight_decay
        )
        return optimizer
    else:
        raise NameError('Can not recognize the name of optimizer')


def step_save_ckpt(path_dict, epoch_id, net, optimizer, logger, args):
    net_ckpt_path = os.path.join(path_dict['net_sub_path'], f"net_{epoch_id}.ckpt")
    torch.save([net.state_dict(), optimizer.state_dict(), logger], net_ckpt_path)


def step_save_label_cache(path_dict, epoch_id, train_set, test_set):
    save_path = os.path.join(path_dict['net_sub_path'], f"labels_cache_{epoch_id}.pt")
    state = {
        'train_dynamic_class': train_set.dynamic_class_labels,
        'train_dynamic_inst': train_set.dynamic_instance_labels,
        'test_dynamic_class': test_set.dynamic_class_labels,
        'test_dynamic_inst': test_set.dynamic_instance_labels
    }
    torch.save(state, save_path)


def step_load_label_cache(path_dict, start_epoch, train_set, test_set):
    if start_epoch < 1:
        return
    load_path = os.path.join(path_dict['net_sub_path'], f"labels_cache_{start_epoch}.pt")
    if os.path.exists(load_path):
        state = torch.load(load_path)
        train_set.dynamic_class_labels = state['train_dynamic_class']
        train_set.dynamic_instance_labels = state['train_dynamic_inst']
        test_set.dynamic_class_labels = state['test_dynamic_class']
        test_set.dynamic_instance_labels = state['test_dynamic_inst']


def step_load_ckpt(path_dict, net, optimizer, logger, args, if_load=False):
    if if_load:
        ckpt_path = os.path.join(
            path_dict['net_sub_path'], f"net_{args.ckpt_load_epoch}.ckpt"
        )
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint[0])
        optimizer.load_state_dict(checkpoint[1])
        logger = checkpoint[2]
        return net, optimizer, logger, args.ckpt_load_epoch
    else:
        return net, optimizer, logger, 0


# ============================================================
# Dataset: ImagePatchDataset
# ============================================================

def get_datasets(datasets_name, train, data_path, args):
    if datasets_name == 'CA1':
        if train:
            dataset_path = os.path.join(data_path, 'train')
        else:
            dataset_path = os.path.join(data_path, 'test')
        dataset = ImagePatchDataset(npz_dir=dataset_path)
        return dataset

class ImagePatchDataset(Dataset):
    def __init__(self, npz_dir):
        self.file_list = glob.glob(os.path.join(npz_dir, "p_*.npz"))
        self.file_list.sort()
        self.dynamic_instance_labels = {}
        self.dynamic_class_labels = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        basename = os.path.basename(path)
        name_parts = basename.split('.')[0].split('_')
        row = int(name_parts[1])
        col = int(name_parts[2])

        with np.load(path) as data:
            image = torch.from_numpy(data['image']).float()
            spots = torch.from_numpy(data['spots']).float()
            dapi = torch.from_numpy(data['dapi']).float()
            raw_label = torch.from_numpy(data['label']).long()
            raw_instance = torch.from_numpy(data['instance']).long()

            fixed_class_label = raw_label + 1
            fixed_instance_label = raw_instance + 1

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
            row,
            col,
            fixed_class_label,
            fixed_instance_label
        )

    def update_label_cache(self, indices, new_class_labels, new_instance_labels):
        for i, idx_tensor in enumerate(indices):
            idx = idx_tensor.item()
            self.dynamic_class_labels[idx] = new_class_labels[i].detach().cpu().clone()
            self.dynamic_instance_labels[idx] = new_instance_labels[i].detach().cpu().clone()


# ============================================================
# Net Utils / UNet
# ============================================================

def get_net(net_name,args):
    if net_name == 'unet':
        net = get_model_unet(args.gpu_id,args.input_channel,args.output_channel)
    else:
        raise NameError('Can not recognize name of net!')
    return net

def get_model_unet(gpu_id, input_channel, output_channel):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    net = UNet(input_channel, output_channel).cuda()
    return net


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

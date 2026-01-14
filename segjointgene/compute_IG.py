import torch
from captum.attr import IntegratedGradients

def setup_ig_solver(net):
    def forward_func_wrapper(inputs, target_class_idx):
        out = net(inputs)
        return out[:, target_class_idx].sum(dim=(1, 2))

    try:
        return IntegratedGradients(forward_func_wrapper)
    except:
        return None

def compute_IG(input_img, net, target_gene, target_celltype):
    ig_solver = setup_ig_solver(net)
    if ig_solver is None:
        return None
    torch.cuda.empty_cache()

    input_tensor = input_img.detach().clone()
    input_tensor.requires_grad = True
    B, C, H, W = input_tensor.shape
    n_celltype = len(target_celltype)
    n_gene = len(target_gene)

    batch_attr = torch.zeros((B, n_celltype, n_gene, H, W), device='cpu')
    for i, c_type in enumerate(target_celltype):
        try:
            attr_all = ig_solver.attribute(
                input_tensor,
                additional_forward_args=(c_type,),
                n_steps=5,                     # <<< 从 20 改回安全值
                internal_batch_size=1          # <<< 关键：防止中间态堆积
            )
            for j, g_idx in enumerate(target_gene):
                if g_idx < C:
                    batch_attr[:, i, j, :, :] = (
                        attr_all[:, g_idx, :, :].detach().cpu()
                    )
            del attr_all
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"[IG Error] Class {c_type}: {e}")
            torch.cuda.empty_cache()

    input_tensor.requires_grad = False
    del input_tensor
    torch.cuda.empty_cache()
    batch_attr = torch.abs(batch_attr)

    return batch_attr
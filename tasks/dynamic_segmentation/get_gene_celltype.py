import os
import torch


def target_gene_celltype(train_loader, args):
    n_gene = int(args.n_gene)
    n_celltype = int(args.n_celltype)
    target_gene = list(range(n_gene))

    if n_celltype == int(args.output_channel):
        target_celltype = list(range(n_celltype))
    else:
        print("\n[Stats] Scanning training set to rank CellTypes...")

        total_class_counts = None
        with torch.no_grad():
            for batch_data in train_loader:
                label = batch_data[1]
                label_flat = label.view(-1).long()
                current_counts = torch.bincount(label_flat)
                if total_class_counts is None:
                    total_class_counts = current_counts
                else:
                    if current_counts.numel() > total_class_counts.numel():
                        new_counts = torch.zeros_like(current_counts)
                        new_counts[: total_class_counts.numel()] = total_class_counts
                        total_class_counts = new_counts
                    total_class_counts[: current_counts.numel()] += current_counts

        total_class_counts = total_class_counts.cpu()
        sorted_classes = torch.argsort(total_class_counts, descending=True).tolist()
        filtered_classes = sorted_classes
        real_class_n = min(n_celltype, len(filtered_classes))
        if real_class_n == 0 and len(sorted_classes) > 0:
            filtered_classes = [0]
            real_class_n = 1
        target_celltype = filtered_classes[:real_class_n]
        print(f"[Config] Grid: {n_gene} Genes x {real_class_n} CellTypes")
        print(f"  Target Genes: {target_gene}")
        print(f"  Target CellTypes: {target_celltype}")
        print(f"  CellTypes Counts: {total_class_counts}")
        total_sum = total_class_counts[1:].sum().item()
        normalized_counts = [1.0] + [round(x, 3) for x in (total_class_counts[1:] / total_sum).tolist()]
        print(f"Normalized List: {normalized_counts}")

    return target_gene, target_celltype


def load_id_name_map(txt_path):
    id2name = {}
    with open(txt_path, "r") as f:
        for line in f:
            idx, name = line.strip().split("\t", 1)
            id2name[str(int(idx))] = name
    return id2name


def get_gene_celltype(data_path: str, args, train_loader):
    """``data_path`` should match ``cfg.dataset.data_path`` (e.g. ``${paths.data_dir}/CA1``)."""
    gene_id_map = load_id_name_map(os.path.join(data_path, "gene_id_map.txt"))
    celltype_id_map = load_id_name_map(os.path.join(data_path, "celltype_id_map.txt"))
    target_gene, target_celltype = target_gene_celltype(train_loader, args)
    target_gene_names = [gene_id_map[str(g)] for g in target_gene]
    target_celltype_names = [celltype_id_map[str(c)] for c in target_celltype]
    print("10 Select genes:", target_gene_names[:10])
    print("10 Select cells:", target_celltype_names[:10])
    return target_gene, target_celltype, target_gene_names, target_celltype_names

"""Map Hydra OmegaConf to a flat namespace for the training loop and get_gene_celltype."""

from __future__ import annotations

from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

from dlbase.training.standard_lightning import early_stopping_patience_epochs


def hydra_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    ds = cfg.dataset
    m = cfg.model
    tr = cfg.train
    ex = cfg.experiment
    sg = cfg.dynamic_seg
    opt = tr.optimizer

    ns = SimpleNamespace()
    ns.datasets_name = str(ds.name)
    ns.net_name = str(m.name)
    ns.net_sub_suffix = str(ex.sub_name)
    ns.random_seed = int(cfg.seed)
    ns.gpu_id = int(sg.gpu_id)
    ns.num_workers = int(ds.num_workers)
    ns.net_batch_size = int(ds.batch_size)
    ns.net_epoch = int(tr.max_epochs)
    ns.net_optimizer = str(opt.name)
    ns.net_lr = float(opt.lr)
    ns.net_weight_decay = float(opt.weight_decay)

    sg_dict = OmegaConf.to_container(sg, resolve=True)
    assert isinstance(sg_dict, dict)
    for k, v in sg_dict.items():
        if k == "predict_epoch":
            continue
        setattr(ns, k, v)

    raw_pe = OmegaConf.select(cfg, "dynamic_seg.predict_epoch", default=None)
    if raw_pe is None:
        ns.predict_epoch = early_stopping_patience_epochs(int(tr.max_epochs))
    else:
        ns.predict_epoch = int(raw_pe)

    ns.input_channel = int(sg.input_channel)
    ns.output_channel = int(sg.output_channel)
    return ns


def hydra_cfg_to_preprocess_args(cfg: DictConfig) -> SimpleNamespace:
    """Flat namespace for ``datasets.preprocess.WMB`` / ``datasets.preprocess.CA1`` / ``datasets.preprocess.Simulation``."""
    ds = cfg.dataset
    sg = cfg.dynamic_seg
    pp = OmegaConf.select(cfg, "dataset.preprocess", default=OmegaConf.create({}))
    pp_dict = OmegaConf.to_container(pp, resolve=True)
    assert isinstance(pp_dict, dict)

    def _pf(key: str, default: float) -> float:
        v = pp_dict.get(key, default)
        return float(v)

    def _pi(key: str, default: int) -> int:
        v = pp_dict.get(key, default)
        return int(v)

    ns = SimpleNamespace()
    ns.datasets_name = str(ds.name)
    if "patch_size" in pp_dict and pp_dict["patch_size"] is not None:
        ns.patch_size = int(pp_dict["patch_size"])
    else:
        ns.patch_size = int(sg.patch_size)
    ns.n_gene = int(sg.n_gene)
    ns.n_celltype = int(sg.n_celltype)
    ns.attr_n_celltype = int(sg.n_celltype)
    ns.visium_grid_size = _pi("visium_grid_size", 4)
    ns.density_sigma = _pf("density_sigma", 4.0)
    ns.global_scale = _pf("global_scale", 1.0)
    ns.max_distance = _pf("max_distance", 15.0)

    ds_container = OmegaConf.to_container(ds, resolve=True)
    if isinstance(ds_container, dict):
        for key in (
            "WMB_sub_path",
            "WMB_label_type",
            "WMB_select_method",
            "WMB_num_celltypes",
            "num_select_genes",
        ):
            if key in ds_container and ds_container[key] is not None:
                setattr(ns, key, ds_container[key])

    pp_dict_check = OmegaConf.to_container(
        OmegaConf.select(cfg, "dataset.preprocess", default=OmegaConf.create({})),
        resolve=True,
    )
    if isinstance(pp_dict_check, dict):
        for sim_key in (
            "Simulation_n_clusters",
            "Simulation_min_large_clusters",
            "Simulation_cluster_min_frac",
            "Simulation_kmeans_k_min",
            "Simulation_kmeans_k_max",
            "Simulation_dapi_close_radius",
            "Simulation_dapi_gaussian_sigma",
            "Simulation_dapi_min_hole_area",
            "Simulation_split_seed",
            "Simulation_train_tenths",
            "Simulation_val_tenths",
        ):
            if sim_key in pp_dict_check and pp_dict_check[sim_key] is not None:
                v = pp_dict_check[sim_key]
                if sim_key == "Simulation_cluster_min_frac":
                    setattr(ns, sim_key, float(v))
                else:
                    setattr(ns, sim_key, int(v))
        if "Protein_cluster_k" in pp_dict_check and pp_dict_check["Protein_cluster_k"] is not None:
            ns.Protein_cluster_k = int(pp_dict_check["Protein_cluster_k"])
        if (
            "Protein_test_roi_indices" in pp_dict_check
            and pp_dict_check["Protein_test_roi_indices"] is not None
        ):
            ns.Protein_test_roi_indices = str(pp_dict_check["Protein_test_roi_indices"])
        if (
            "Protein_exclude_prefix_count" in pp_dict_check
            and pp_dict_check["Protein_exclude_prefix_count"] is not None
        ):
            ns.Protein_exclude_prefix_count = int(pp_dict_check["Protein_exclude_prefix_count"])
        if (
            "Protein_test_tail_count" in pp_dict_check
            and pp_dict_check["Protein_test_tail_count"] is not None
        ):
            ns.Protein_test_tail_count = int(pp_dict_check["Protein_test_tail_count"])

    return ns

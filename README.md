# SegJointGene-CID (Simulation Standalone)

## Overview

SegJointGene is a self-training framework for spatial cell-type segmentation that integrates **Computational Information Discarding (CID)** to constrain iterative label propagation. The method combines a segmentation network with attribution-guided label updates to progressively refine pixel-wise class and instance labels.

This folder is a **standalone, runnable package** for:

- **Dataset:** Simulation
- **Model:** UNet
- **Task:** `dynamic_segmentation_CID`

Directory layout mirrors the main repository (`main.py`, `dlbase/`, `datasets/`, `models/`, `tasks/`, `conf/`).

---

## 1. Environment & Installation

### Python Version

This codebase is tested with:

- **Python 3.11.8** (also works with 3.10+ / 3.12 in practice)

### Dependencies

Typical requirements include:

- `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`
- `hydra-core`, `omegaconf`, `pytorch-lightning`, `tensorboard`
- `scikit-learn`, `scikit-image`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `joblib`

#### pip install

```bash
cd code_simulation_cid
python3 -m pip install -r requirements.txt
python3 scripts/check_env.py
```

#### conda install (optional)

```bash
conda create -n SegJointGene python=3.11 -y
conda activate SegJointGene
conda install pytorch pytorch-cuda=12.1 numpy scipy pandas matplotlib opencv scikit-image tifffile -c pytorch -c nvidia -c conda-forge
python3 -m pip install hydra-core omegaconf pytorch-lightning tensorboard tqdm joblib scikit-learn
```

---

## 2. Algorithm Overview

SegJointGene-CID follows an iterative self-training paradigm with attribution-based constraints:

1. **Segmentation Network**  
   A UNet-style network predicts pixel-wise cell-type labels for each image patch.

2. **Self-Training with Dynamic Labels**  
   Instead of using fixed ground-truth labels only, the dataset maintains **dynamic labels** updated after each iteration.

3. **CID Attribution**  
   For selected cell types and genes, CID computes pixel-wise attribution by optimizing an input noise mask while freezing network weights. Pixels that tolerate larger noise are considered less informative.

4. **Attribution-Constrained Label Update**  
   During label propagation:
   - Predictions must satisfy spatial consistency and confidence constraints.
   - The dominant attribution class must match the propagated class.  
   This suppresses spurious label expansion and stabilizes self-training.

5. **Iterative Refinement**  
   The process repeats over epochs, progressively improving segmentation performance.

---

## 3. Dataset Interface

The framework uses a patch-based dataset; each sample is stored as a `.npz` file.

### Dataset Class

`datasets/Simulation/Simulation.py :: ImagePatchDataset` supports:

- Immutable fixed labels (used as a core mask)
- Mutable dynamic labels (updated during self-training)
- Persistent label caching across epochs

Dynamic labels are automatically loaded and saved during training.

### Required `.npz` File Format

Each patch file must contain:

| Key | Shape / Type | Description |
|-----|--------------|-------------|
| `image` | `(C, H, W)` float32 | Input image (gene expression channels) |
| `label` | `(H, W)` int | Initial class label map |
| `instance` | `(H, W)` int | Initial instance label map |
| `spots` | `(H, W)` float32 | Spot density or auxiliary spatial signal |
| `dapi` | `(H, W)` float32 | DAPI or reference channel |
| `ground_truth` | `(H, W)` int | Optional ground-truth map |

### File Naming

```
p_<group_row>_<group_col>_<row>_<col>.npz
```

`<row>` and `<col>` indicate the spatial grid position of the patch.

---

## 4. Preprocess Demo Dataset

### Simulation raw layout

Place raw Simulation data under:

```
data/Simulation_raw/
  image_*.png
  label_*.png
  spots_*.csv
```

After preprocessing, outputs are written to:

```
data/Simulation/{train,val,test}/p_*.npz
data/Simulation_image/
data/Simulation/n_celltype.txt
```

### Preprocess command

```bash
# Skip if outputs already exist
bash scripts/preprocess_simulation.sh ./data

# Force rebuild
bash scripts/preprocess_simulation.sh ./data --force

# Python equivalent
python3 scripts/preprocess_simulation.py --data_dir ./data
python3 scripts/preprocess_simulation.py --data_dir ./data --force
```

If you already have preprocessed NPZ under `data/Simulation/`, you can skip this step.

---

## 5. Running SegJointGene-CID

### Basic Command

The main entry point is `main.py`:

```bash
cd code_simulation_cid
python3 main.py --sub_path basic
```

This will:

- Initialize the UNet segmentation network
- Load patch-based data from `data/Simulation/`
- Run iterative self-training with CID attribution
- Automatically manage dynamic label caching and checkpoints

Equivalent helper script:

```bash
bash scripts/run_cid.sh ./data basic 100
```

### Common Hydra Overrides

| Argument | Description | Default (Simulation) |
|----------|-------------|----------------------|
| `paths.data_dir` | Data root directory | `./data` |
| `train.max_epochs` / `--net_epoch` | Training epochs | `100` |
| `dynamic_seg.patch_size` | Patch resolution | `128` |
| `dynamic_seg.attr_epoch` | Epoch to start CID attribution | `0` |
| `dynamic_seg.n_gene` | Number of target genes (channels) | `980` |
| `dynamic_seg.n_celltype` | Number of cell types (incl. background) | `6` |
| `dynamic_seg.CID_n_steps` | Optimization steps for CID | `20` |
| `dynamic_seg.CID_gene_chunk_size` | Genes processed per CID chunk | `100` |
| `dynamic_seg.CID_noise_num` | Monte Carlo noise samples | `8` |
| `dynamic_seg.if_load_ckpt` | Resume from checkpoint | `false` |
| `dynamic_seg.ckpt_load_epoch` | Checkpoint epoch to load | `30` |
| `dataset.rerun_preprocess` / `--rerun_preprocess` | Rebuild NPZ before training | `false` |

Example with custom CID start epoch:

```bash
python3 main.py --sub_path basic --net_epoch 100 \
  paths.data_dir=/path/to/data \
  dynamic_seg.attr_epoch=50 \
  dynamic_seg.CID_n_steps=50
```

---

## 6. Output

During execution, the framework automatically:

- Updates dynamic labels in memory
- Saves label caches (`labels_cache_<epoch>.pt`)
- Saves model checkpoints (`epoch_<n>.ckpt`)
- Writes TensorBoard logs under `tensorboard/`
- Saves visualization panels and CID heatmaps under `visualize/epoch_<n>/`
- Writes `summary.log` (when enabled)

Default experiment output directory:

```
experiment/Simulation/unet/dynamic_segmentation_CID/<sub_path>_<seed>/
```

Example:

```
experiment/Simulation/unet/dynamic_segmentation_CID/basic_1/
```

---

## 7. Summary

SegJointGene-CID provides a minimal yet expressive framework for attribution-guided self-training in spatial segmentation, combining:

- Patch-based segmentation
- Dynamic label propagation
- Computational Information Discarding guided training

This standalone package focuses on the **Simulation + UNet + CID** workflow and can be distributed independently of the full multi-dataset repository.

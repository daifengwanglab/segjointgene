## Overview

SegJointGene is a self-training framework for spatial cell-type segmentation that integrates **Computational Information Discarding (CID)** to constrain iterative label propagation.
The method combines a segmentation network with attribution-guided label updates to progressively refine pixel-wise class and instance labels.

---

## 1. Environment & Installation

### Python Version

This codebase is tested with:

Python 3.11.8

### Dependencies

Typical requirements include:

* `torch`
* `numpy`
* `argparse`
* `cv2`
* `captum`
* `skimage`
* `scipy`

You can install by conda:

* `conda create -n segjointgene python=3.11 -y`
* `conda activate segjointgene`
* `conda install pytorch numpy opencv scikit-image scipy captum -c pytorch -c conda-forge`

---

## 2. Algorithm Overview

SegJointGene-CID follows an **iterative self-training paradigm** with attribution-based constraints:

1. **Segmentation Network**
   A UNet-style network predicts pixel-wise cell-type labels for each image patch.

2. **Self-Training with Dynamic Labels**
   Instead of using fixed ground-truth labels, the dataset maintains **dynamic labels** that are updated after each iteration.

3. **CID Attribution**
   For selected cell types and genes, CID computes pixel-wise attribution by optimizing an input noise mask while freezing network weights.
   Pixels that tolerate larger noise are considered less informative.

4. **Attribution-Constrained Label Update**
   During label propagation:

   * Predictions must satisfy spatial consistency and confidence constraints.
   * The dominant attribution class must match the propagated class.
     This suppresses spurious label expansion and stabilizes self-training.

5. **Iterative Refinement**
   The process repeats over epochs, progressively improving segmentation performance.

---

## 3. Dataset Interface

The framework uses a **patch-based dataset**, where each sample is stored as a `.npz` file.

### Dataset Class

The dataset class (`ImagePatchDataset`) supports:

* Immutable **fixed labels** (used as a core mask)
* Mutable **dynamic labels** (updated during self-training)
* Persistent label caching across epochs

Dynamic labels are automatically loaded and saved during training.

---

### Required `.npz` File Format

Each patch file **must** contain the following keys:

| Key        | Shape / Type        | Description                                 |
| ---------- | ------------------- | ------------------------------------------- |
| `image`    | `(C, H, W)` float32 | Input image (e.g. gene expression channels) |
| `label`    | `(H, W)` int        | Initial class label map                     |
| `instance` | `(H, W)` int        | Initial instance label map                  |
| `spots`    | `(H, W)` float32    | Spot density or auxiliary spatial signal    |
| `dapi`     | `(H, W)` float32    | DAPI or reference channel                   |

File naming must follow:

p_<row>_<col>.npz

where `<row>` and `<col>` indicate the spatial grid position of the patch.

---

## 4. Preprocess demo dataset dataset

One section from the mouse hippocumpus dataset is in `data/CA1_raw/3_1_left`, from [Probabilistic cell typing enables fine mapping of closely related cell types in situ](https://www.nature.com/articles/s41592-019-0631-4#data-availability).
The paper provided a link to download full dataset: [mouse hippocumpus CA1 region](https://figshare.com/articles/dataset/CA1_neuron_cell_typing_results_of_pciSeq_probabilistic_cell_typing_by_in_situ_sequencing_/7150760).

To preprocess this dataset and start training, run:

* `python main.py --datasets_name=CA1 --step_name=preprocess_CA1 --CA1_sub_path=3_1_left`

## 5. Running SegJointGene

### Basic Command

The main entry point is `main.py`.
To run the CID-based self-training pipeline:

* `python main.py --datasets_name=CA1 --step_name=SegjointGene --attr_method=CID`

This will:

* Initialize the segmentation network
* Load patch-based data from `data/CA1/`
* Run iterative self-training with CID attribution
* Automatically manage dynamic label caching and checkpoints

---

### Commonly Used Arguments

| Argument           | Description                                         |
| ------------------ | --------------------------------------------------- |
| `--patch_size`     | Patch resolution                                    |
| `--attr_epoch`     | Epoch to start CID attribution                      |
| `--attr_n_gene`      | Number of target genes for attribution |
| `--attr_n_celltype`      | Number of target cell types for attribution |
| `--CID_n_steps`    | Optimization steps for CID                          |
| `--CID_chunk_size` | Number of cell types processed per CID chunk        |
| `--if_load_ckpt`   | Resume from a saved checkpoint                      |

All arguments are defined in `main.py`.

---

## 6. Output

During execution, the framework will automatically:

* Update dynamic labels in memory
* Save label caches at regular intervals
* Save model checkpoints

---

## 7. Summary

SegJointGene provides a minimal yet expressive framework for **attribution-guided self-training in spatial segmentation**, combining:

* Patch-based segmentation
* Dynamic label propagation
* Computational Information Discarding guided training
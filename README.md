# CT-ScanGaze: A Dataset and Baselines for 3D Volumetric Scanpath Modeling

<div align="center">

[![ICCV 2025](https://img.shields.io/badge/ICCV%202025-Highlight-orange.svg)](https://iccv2025.thecvf.com/)
[![Paper](https://img.shields.io/badge/arXiv-2507.12591-red.svg)](https://arxiv.org/html/2507.12591v1)
[![Dataset](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/phamtrongthang/CT-ScanGaze)
[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-yellow.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

</div>

This repository provides the first publicly available dataset of expert radiologist gaze during CT analysis **CTScanGaze** and **CT-Searcher**, a transformer-based model for 3D scanpath prediction on medical CT volumes. Our work addresses the critical gap in understanding how radiologists visually examine 3D medical images during diagnostic procedures.

<div align="center">
  <img src="docs/image.png" alt="CT-ScanGaze Dataset Overview" width="80%">
  <br>
  <em><strong>Figure 1:</strong> CTScanGaze</em>
</div>

<div align="center">
  <img src="docs/image-1.png" alt="CT-Searcher Model Architecture" width="80%">
  <br>
  <em><strong>Figure 2:</strong> CTSearcher</em>
</div>

**🎉 This work has been accepted as a highlight paper at ICCV 2025!**

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Contact](#contact)

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/UARK-AICV/CTScanGaze
cd CTScanGaze

# Create conda environment
conda create -n ctsearcher python=3.9
conda activate ctsearcher

pip install uv 

uv pip install -r requirements.txt
```


## Dataset

**CT-ScanGaze** is the first publicly available eye gaze dataset focused on CT scan analysis. The dataset is available on [Hugging Face](https://huggingface.co/datasets/phamtrongthang/CT-ScanGaze).


Each data sample contains the following fields:

```python
{
    "name": str,           # CT scan identifier
    "subject": str,        # Radiologist ID
    "task": str,           # Task description
    "X": list,             # X coordinates of fixations
    "Y": list,             # Y coordinates of fixations
    "Z": list,             # Z coordinates (slice numbers)
    "T": list,             # Fixation durations in seconds
    "length": int,         # Scanpath length
    "split": str,          # Data split ("train" or "test")
    "report": str,         # Report for this CT
}
```
Note that other fields in the JSON are dummy, so you do not need to care about them. For the reports, many reports will look like duplications because multiple CTs are from the same CT reading session for the same patient.

Additionally, we provide zip files containing all CT scans that match the identifiers, along with corresponding radiological reports for each CT scan.


## Feature Extraction

Before training, you need to extract Swin UNETR features from your CT volumes. We provide a two-step process:

### Step 1: Extract Features

Extract features from CT volumes using a pre-trained Swin UNETR model:

```bash
# Place your CT volumes in one_sample/cts/*.nii.gz
uv run feature_extraction/swin_unet_extract_feature.py
```

This script will:
- Download the pre-trained Swin UNETR model (MONAI BTCV weights)
- Extract features using sliding window (96×96×96 patches)
- Save patch-based features to `one_sample/features/*.pt`

### Step 2: Merge Features

Merge overlapping patch features into complete volumes:

```bash
python feature_extraction/merge_features.py \
    --features_dir one_sample/features \
    --output_dir one_sample/features_merged
```

This creates final feature volumes:
- `{name}.pt`: Decoder features (768 channels, H/32×W/32×D/32)
- `{name}_hidden_states_out_4.pt`: Encoder features (768 channels, H/32×W/32×D/32)

For more details, see [feature_extraction/README.md](feature_extraction/README.md).

## Training

### Quick Start

After extracting features, you can train the CT-Searcher model:

#### Local Training
```bash
# Single or multi-GPU
bash bash/train.sh

# Or directly
python src/train_lightning.py \
    --log_root runs/experiment \
    --epoch 40 \
    --batch 2 \
    --img_dir /path/to/data \
    --feat_dir /path/to/features_merged
```

#### Slurm Cluster (Multi-node)
```bash
sbatch bash/train_slurm.sh
```

Lightning auto-detects Slurm and configures multi-node DDP. Adjust `--nodes` and `--gres=gpu:X` in the script as needed.

**Features:**
- Auto multi-GPU/multi-node training
- Mixed precision (16-bit)
- Smart checkpointing
- TensorBoard logging
- Slurm auto-detection

### Resume Training

```bash
python src/train_lightning.py \
    --resume_dir runs/experiment_name \
    --batch 2 \
    --epoch 40
```

The trainer will automatically load the last checkpoint from the specified directory.

## Evaluation

### Test a Trained Model

Evaluation is performed automatically during training (every epoch). To evaluate a saved checkpoint:

```bash
python src/train_lightning.py \
    --resume_dir runs/CTScanGaze_CTSearcher \
    --img_dir /path/to/test/ct/images \
    --feat_dir /path/to/test/features \
    --fix_dir /path/to/test/gaze/data
```

The Lightning trainer handles validation automatically with comprehensive metrics.

### Evaluation Metrics

We use comprehensive 3D-adapted metrics for scanpath evaluation:

**Scanpath-based Metrics:**
- **ScanMatch (SM)**: Spatial and temporal similarity with duration consideration
- **MultiMatch (MM)**: Five-dimensional assessment (shape, direction, length, position, duration)
- **String Edit Distance (SED)**: Sequence-based comparison using Levenshtein distance

**Spatial-based Metrics:**
- **Correlation Coefficient (CC)**: Linear correlation between predicted and ground truth heatmaps
- **Normalized Scanpath Saliency (NSS)**: Normalized saliency at fixation locations
- **Kullback-Leibler Divergence (KLDiv)**: Distribution similarity measure

## TODO

The current code base is working as long as the path and extracted features are prepared. But a lot of refactoring work is needed.

- [x] Extracted feature of CTs (see [Feature Extraction](#feature-extraction))
- [ ] Clean and refactor codebase
- [ ] Synthetic dataset
- [ ] Improve code comments and structure


## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{pham2025ct,
  title={CT-ScanGaze: A Dataset and Baselines for 3D Volumetric Scanpath Modeling},
  author={Pham, Trong-Thang and Awasthi, Akash and Khan, Saba and Marti, Esteban Duran and Nguyen, Tien-Phat and Vo, Khoa and Tran, Minh and Nguyen, Ngoc Son and Van, Cuong Tran and Ikebe, Yuki and others},
  journal={arXiv preprint arXiv:2507.12591},
  year={2025}
}
```

## Acknowledgments

This material is based upon work supported by the National Science Foundation (NSF) under Award No OIA-1946391, NSF 2223793 EFRI BRAID, National Institutes of Health (NIH) 1R01CA277739-01.

## License

This project is licensed under the Creative Commons Attribution Non Commercial Share Alike 4.0 International License. See the [LICENSE](LICENSE) file for details.



## Contact

**Primary Contact**: Trong Thang Pham (tp030@uark.edu)

For questions, feedback, or collaboration opportunities, feel free to reach out! I would love to hear from you if you have any thoughts or suggestions about this work.

**Note**: While we don't actively seek contributions to the codebase, we greatly appreciate and welcome feedback, discussions, and suggestions for improvements.


---

<div align="center">

**Star this repository if you find it useful!**

</div>

# Feature Extraction for CT-Searcher

This directory contains scripts for extracting Swin UNETR features from CT volumes.

## Quick Start

For end-to-end feature extraction and merging:

```bash
# Extract features from CT volumes (creates patch-based features)
uv run feature_extraction/swin_unet_extract_feature.py

# Merge overlapping patches into complete feature volumes
uv run feature_extraction/merge_features.py \
    --features_dir one_sample/features \
    --output_dir one_sample/features_merged
```

## Two-Step Process

### Step 1: Extract Features

Extracts Swin UNETR features using sliding window approach:

```bash
uv run feature_extraction/swin_unet_extract_feature.py
```

**Input**: CT volumes in `one_sample/cts/*.nii.gz`
**Output**: Patch-based features in `one_sample/features/*.pt`

Each `.pt` file contains a list of patch dictionaries with:
- `hidden_states_out`: Encoder output features (768-dim)
- `dec4`: Decoder layer 4 features (768-dim)
- `unravel_slice`: Patch location metadata

### Step 2: Merge Features

Merges overlapping patch features into complete volumes:

```bash
python feature_extraction/merge_features.py \
    --features_dir one_sample/features \
    --output_dir one_sample/features_merged
```

**Options**:
- `--features_dir`: Directory containing extracted `.pt` files
- `--output_dir`: Directory to save merged features
- `--pattern`: Glob pattern for feature files (default: `*.pt`)
- `--downsample_factor`: Spatial downsampling factor (default: 32)
- `--strict`: Raise error on first failure instead of logging

**Output**: Two files per CT volume:
- `{name}.pt`: Merged dec4 features (768, H/32, W/32, D/32)
- `{name}_hidden_states_out_4.pt`: Merged encoder features (768, H/32, W/32, D/32)

## Technical Details

- **Sliding Window**: Features are extracted in 96×96×96 patches with overlap
- **Downsampling**: Swin UNETR reduces spatial dimensions by 32× (96→3)
- **Merging**: Overlapping regions are averaged for smooth feature transitions
- **Model**: Pre-trained Swin UNETR from MONAI (BTCV segmentation weights)

## For Training

The CT-Searcher model expects merged features. Set `--feat_dir` to point to the merged features directory:

```bash
python src/train_lightning.py \
    --feat_dir one_sample/features_merged \
    --img_dir one_sample/cts \
    ...
```
#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     "tqdm",
#   "numpy"
# ]
# ///
"""
Merge overlapping patch features from Swin UNETR feature extraction.

This script takes the sliding window features extracted from swin_unet_extract_feature.py
and merges them into complete feature volumes by averaging overlapping regions.

The features are extracted in patches (e.g., 96x96x96) with sliding window, and this
script stitches them back together into the full volume representation.
"""

import json
import os
import warnings
from pathlib import Path

import torch
from tqdm import tqdm


def get_all_file_inside(directory, pattern="*.pt"):
    """Recursively get all files matching pattern inside directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    all_files = list(directory_path.rglob(pattern))
    return sorted(all_files)


def makedirs_parent(file_path):
    """Create parent directories for a file path if they don't exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def save_as_json(data, file_path):
    """Save data as JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def merge_swin_features(
    features_dir,
    output_dir,
    feature_pattern="*.pt",
    downsample_factor=32,
    ignore_errors=True,
):
    """
    Merge overlapping Swin UNETR feature patches into complete volumes.

    Args:
        features_dir: Directory containing extracted feature files (.pt)
        output_dir: Directory to save merged features
        feature_pattern: Glob pattern to match feature files (default: "*.pt")
        downsample_factor: Spatial downsampling factor from original volume (default: 32)
                          This is because Swin UNETR reduces 96x96x96 patches to 3x3x3 features
        ignore_errors: If True, skip files that fail to process and log them

    Returns:
        List of files that failed to process (if ignore_errors=True)
    """
    # Find all feature files
    all_features = get_all_file_inside(features_dir, pattern=feature_pattern)

    if not all_features:
        raise ValueError(
            f"No feature files found in {features_dir} matching {feature_pattern}"
        )

    print(f"Found {len(all_features)} feature files to merge")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    failed_files = []
    pbar = tqdm(all_features, desc="Merging features")

    for feature_path in pbar:
        try:
            # Load feature file (list of dicts with keys: hidden_states_out, dec4, unravel_slice)
            # Note: weights_only=False is safe here as we control the feature extraction process
            features = torch.load(feature_path, weights_only=False)

            if not isinstance(features, list):
                warnings.warn(f"Unexpected feature format in {feature_path}, skipping")
                failed_files.append(str(feature_path))
                continue

            # Determine the bounds of all patches to create output volume
            min_w, min_h, min_c = 999, 999, 999
            max_w, max_h, max_c = 0, 0, 0

            for patch_data in features:
                if "unravel_slice" not in patch_data:
                    warnings.warn(f"Missing unravel_slice in {feature_path}, skipping")
                    failed_files.append(str(feature_path))
                    break

                for sl in patch_data["unravel_slice"]:
                    # Each slice is [batch_slice, channel_slice, h_slice, w_slice, c_slice]
                    h_sl = sl[2]
                    w_sl = sl[3]
                    c_sl = sl[4]

                    min_w = min(min_w, w_sl.start)
                    min_h = min(min_h, h_sl.start)
                    min_c = min(min_c, c_sl.start)
                    max_w = max(max_w, w_sl.stop)
                    max_h = max(max_h, h_sl.stop)
                    max_c = max(max_c, c_sl.stop)

            # Calculate output dimensions (downsampled by factor)
            h32 = max_h // downsample_factor
            w32 = max_w // downsample_factor
            c32 = max_c // downsample_factor

            # Initialize output tensors
            # Count map tracks how many patches contribute to each voxel (for averaging)
            count_map_list = torch.zeros([1, 1, h32, w32, c32])

            # Initialize feature tensors (768 channels from Swin UNETR)
            hidden_states_out = torch.zeros((768, h32, w32, c32))
            dec4 = torch.zeros_like(hidden_states_out)

            # Accumulate features from all patches
            for feat in features:
                # Extract patch features
                hidden_states_out_i = feat.get("hidden_states_out")
                dec4_i = feat.get("dec4")
                unravel_slice_i = feat["unravel_slice"]

                # Process hidden_states_out (main encoder output)
                if hidden_states_out_i is not None:
                    # Handle both single tensor and list of tensors
                    if isinstance(hidden_states_out_i, list):
                        hidden_states_out_i = hidden_states_out_i[-1]  # Take last layer

                    for original_idx, patch in zip(
                        unravel_slice_i, hidden_states_out_i
                    ):
                        # Convert original volume indices to downsampled feature indices
                        idx_zm = list(original_idx)
                        new_idx = tuple(
                            [
                                idx_zm[1],  # Channel index
                                slice(
                                    idx_zm[2].start // downsample_factor,
                                    idx_zm[2].stop // downsample_factor,
                                ),
                                slice(
                                    idx_zm[3].start // downsample_factor,
                                    idx_zm[3].stop // downsample_factor,
                                ),
                                slice(
                                    idx_zm[4].start // downsample_factor,
                                    idx_zm[4].stop // downsample_factor,
                                ),
                            ]
                        )
                        hidden_states_out[new_idx] += patch

                # Process dec4 (decoder output)
                if dec4_i is not None:
                    for original_idx, patch in zip(unravel_slice_i, dec4_i):
                        idx_zm = list(original_idx)
                        new_idx = tuple(
                            [
                                idx_zm[1],
                                slice(
                                    idx_zm[2].start // downsample_factor,
                                    idx_zm[2].stop // downsample_factor,
                                ),
                                slice(
                                    idx_zm[3].start // downsample_factor,
                                    idx_zm[3].stop // downsample_factor,
                                ),
                                slice(
                                    idx_zm[4].start // downsample_factor,
                                    idx_zm[4].stop // downsample_factor,
                                ),
                            ]
                        )
                        dec4[new_idx] += patch

                # Update count map
                for original_idx in unravel_slice_i:
                    idx_zm = list(original_idx)
                    new_idx = tuple(
                        [
                            idx_zm[0],  # Batch index
                            idx_zm[1],  # Channel index
                            slice(
                                idx_zm[2].start // downsample_factor,
                                idx_zm[2].stop // downsample_factor,
                            ),
                            slice(
                                idx_zm[3].start // downsample_factor,
                                idx_zm[3].stop // downsample_factor,
                            ),
                            slice(
                                idx_zm[4].start // downsample_factor,
                                idx_zm[4].stop // downsample_factor,
                            ),
                        ]
                    )
                    count_map_list[new_idx] += 1

            # Average overlapping regions (avoid division by zero)
            count_map_list[count_map_list == 0] = 1  # Prevent division by zero
            dec4 /= count_map_list[0]
            hidden_states_out /= count_map_list[0]

            # Determine output file paths
            feature_name = feature_path.name
            if feature_name.endswith(".nii.gz.pt"):
                base_name = feature_name.replace(".nii.gz.pt", ".pt")
                hidden_name = feature_name.replace(
                    ".nii.gz.pt", "_hidden_states_out_4.pt"
                )
            else:
                base_name = feature_name
                hidden_name = feature_name.replace(".pt", "_hidden_states_out_4.pt")

            # Preserve subdirectory structure
            relative_path = feature_path.relative_to(features_dir)
            output_subdir = output_dir / relative_path.parent

            dec4_output_path = output_subdir / base_name
            hidden_output_path = output_subdir / hidden_name

            # Create output directories
            makedirs_parent(dec4_output_path)
            makedirs_parent(hidden_output_path)

            # Save merged features
            torch.save(dec4, dec4_output_path)
            torch.save(hidden_states_out, hidden_output_path)

            pbar.set_postfix_str(f"Saved {base_name}")

        except Exception as e:
            error_msg = f"Error processing {feature_path}: {str(e)}"
            if ignore_errors:
                warnings.warn(error_msg)
                failed_files.append(str(feature_path))
            else:
                raise RuntimeError(error_msg) from e

    # Save log of failed files
    if failed_files:
        log_file = Path(output_dir) / "failed_merges.json"
        save_as_json(failed_files, log_file)
        print(f"\n{len(failed_files)} files failed to process. See {log_file}")
    else:
        print("\nAll files processed successfully!")

    return failed_files


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge overlapping Swin UNETR feature patches into complete volumes"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Directory containing extracted feature files (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save merged features",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="Glob pattern to match feature files (default: *.pt)",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=32,
        help="Spatial downsampling factor from original volume (default: 32)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error on first failure instead of logging and continuing",
    )

    args = parser.parse_args()

    failed_files = merge_swin_features(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        feature_pattern=args.pattern,
        downsample_factor=args.downsample_factor,
        ignore_errors=not args.strict,
    )

    if failed_files:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())

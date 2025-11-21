# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     "monai>=1.3.0",
#     "numpy",
#     "nibabel",
#     "tqdm",
#       "einops"
# ]
# ///
#

import gc
import json
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Add the random_script directory to path FIRST to import encoder module
sys.path.insert(0, str(Path(__file__).parent))
import torch
from encoder.nets.swin_unetr import SwinUNETR
from encoder.utils import sliding_window_encode
from monai.config import print_config
from monai.data import CacheDataset, ThreadDataLoader, set_track_meta
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
)
from tqdm import tqdm


def download_pretrained_model(model_dir="pretrained_models"):
    """Download the Swin UNETR BTCV segmentation model from Hugging Face."""
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)

    model_file = model_path / "model.pt"

    if model_file.exists():
        print(f"Model already exists at {model_file}")
        return str(model_file)

    print("Downloading Swin UNETR BTCV segmentation model from Hugging Face...")
    print("This may take a few minutes (file size ~240MB)...")

    # Direct download URL from Hugging Face
    url = "https://huggingface.co/MONAI/swin_unetr_btcv_segmentation/resolve/main/models/model.pt"

    try:
        print(f"Downloading from {url}...")
        urlretrieve(url, model_file)
        print(f"Model downloaded successfully to {model_file}")
        return str(model_file)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("\nManual download instructions:")
        print(
            "1. Visit: https://huggingface.co/MONAI/swin_unetr_btcv_segmentation/tree/main"
        )
        print("2. Download models/model.pt")
        print(f"3. Save it to: {model_file}")
        raise


def main():
    print_config()

    # Setup paths
    script_dir = Path(__file__).parent.parent
    one_sample_dir = script_dir / "one_sample"
    cts_dir = one_sample_dir / "cts"
    output_dir = one_sample_dir / "features"
    output_dir.mkdir(exist_ok=True)

    # Download pretrained model
    model_file = download_pretrained_model()

    # Setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup transforms (same as script 11)
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )

    # Find all .nii.gz files in the one_sample/cts directory
    ct_files = list(cts_dir.glob("*.nii.gz"))

    if not ct_files:
        print(f"No .nii.gz files found in {cts_dir}")
        return

    print(f"Found {len(ct_files)} CT file(s) to process:")
    for f in ct_files:
        print(f"  - {f.name}")

    # Prepare data list for MONAI dataset
    val_files = [{"image": str(f)} for f in ct_files]

    # Create dataset and dataloader
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    # Disable metadata tracking for training (as in script 11)
    set_track_meta(False)

    # Load Swin UNETR model (matching script 11 parameters)
    print("Loading Swin UNETR model...")
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=14,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    # Load pretrained weights (matching script 11 loading approach)
    model.load_state_dict(torch.load(model_file, weights_only=False))
    model.eval()
    print("Model loaded successfully")

    # Extract features
    with torch.no_grad():
        for case_num in tqdm(range(len(val_ds)), desc="Extracting features"):
            # Get image metadata
            img_path = val_ds[case_num]["image"].meta["filename_or_obj"]
            img_name = Path(img_path).name.split('.')[0]
            output_file = output_dir / f"{img_name}.pt"

            # Skip if already processed
            if output_file.exists():
                print(f"Skipping {img_name} (already processed)")
                continue

            print(f"Processing {img_name}...")

            # Get image and add batch dimension
            img = val_ds[case_num]["image"]
            val_inputs = torch.unsqueeze(img, 1).to(device)

            # Extract features using sliding window
            saved_embeddings = sliding_window_encode(
                val_inputs,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.0,
            )

            # Save features
            torch.save(saved_embeddings, output_file)
            print(f"Saved features to {output_file}")

            # Free up GPU memory
            del val_inputs, saved_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nFeature extraction complete! Features saved to {output_dir}")

    # Save a JSON file with feature paths
    feature_paths = [str(f) for f in output_dir.glob("*.pt")]
    feature_list_file = one_sample_dir / "feature_paths.json"
    with open(feature_list_file, "w") as f:
        json.dump(feature_paths, f, indent=4)
    print(f"Feature paths saved to {feature_list_file}")


if __name__ == "__main__":
    main()

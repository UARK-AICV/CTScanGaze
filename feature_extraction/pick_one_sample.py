# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
#
# This script will go into the ../data folder, read the mds_report.json file, then keep only "3_5_194" as the data.

# Then this script create a one_sample directory.

# Then this script go into ../data/cts/ to copy out the 3_5_194 nii gz out into cts/ subfolder of "one_sample"
# this script also shrink down mds report json to only one item about 3_5_194

import json
import shutil
from pathlib import Path

# Define paths
data_dir = Path(__file__).parent.parent / "data"
mds_report_path = data_dir / "mds_report.json"
cts_dir = data_dir / "cts"
one_sample_dir = Path(__file__).parent.parent / "one_sample"

# Read the mds_report.json file
print(f"Reading {mds_report_path}...")
with open(mds_report_path, "r") as f:
    mds_data = json.load(f)

# Filter to keep only "3_5_194" data
filtered_data = {}
target_name = "3_5_194"

for subject_id, subject_data in mds_data.items():
    filtered_items = [item for item in subject_data if item.get("name") == target_name]
    if filtered_items:
        filtered_data[subject_id] = filtered_items

print(
    f"Filtered data contains {sum(len(v) for v in filtered_data.values())} entries for '{target_name}'"
)

# Create one_sample directory structure
one_sample_dir.mkdir(exist_ok=True)
one_sample_cts_dir = one_sample_dir / "cts"
one_sample_cts_dir.mkdir(exist_ok=True)

print(f"Created directory: {one_sample_dir}")
print(f"Created directory: {one_sample_cts_dir}")

# Copy the 3_5_194.nii.gz file from ../data/cts/ to one_sample/cts/
source_ct_file = cts_dir / f"{target_name}.nii.gz"
dest_ct_file = one_sample_cts_dir / f"{target_name}.nii.gz"

if source_ct_file.exists():
    print(f"Copying {source_ct_file} to {dest_ct_file}...")
    shutil.copy2(source_ct_file, dest_ct_file)
    print("Successfully copied CT file")
else:
    print(f"Warning: Source file {source_ct_file} not found!")

# Save the filtered mds_report.json to one_sample directory
filtered_report_path = one_sample_dir / "mds_report.json"
print(f"Saving filtered report to {filtered_report_path}...")
with open(filtered_report_path, "w") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Successfully created one_sample dataset with '{target_name}'")
print(f"Location: {one_sample_dir}")

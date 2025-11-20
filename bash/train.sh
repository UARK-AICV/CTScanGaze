# The name of this experiment.
DATASET_NAME='CTScanGaze'
MODEL_NAME='CTSearcher'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

# Train with PyTorch Lightning
CUDA_VISIBLE_DEVICES=1,2 python src/train_lightning.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} \
  --epoch 40 \
  --warmup_epoch 1 \
  --batch 2 \
  --img_dir /path/to/CTScanGaze/data \
  --feat_dir /path/to/CTScanGaze/features/swin_unetr_feature 

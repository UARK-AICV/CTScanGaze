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
CUDA_VISIBLE_DEVICES=0 python src/train_lightning.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} \
  --epoch 5 \
  --warmup_epoch 1 \
  --batch 1 \
  --img_dir /home/tp030/CTScanGaze/one_sample \
  --feat_dir /home/tp030/CTScanGaze/one_sample/features_merged \
  --fix_dir /home/tp030/CTScanGaze/one_sample

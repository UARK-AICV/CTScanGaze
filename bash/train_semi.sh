# The name of this experiment.
DATASET_NAME='CTScanGaze'
MODEL_NAME='CTSearcher_semi'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} --epoch 40 --start_rl_epoch 40 --no_eval_epoch 40 --batch 8 --img_dir /path/to/CTScanGaze/semi_data --feat_dir /path/to/CTScanGaze/features/swin_unetr_feature 

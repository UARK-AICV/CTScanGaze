# The name of this experiment.
DATASET_NAME='COCO_Search'
MODEL_NAME='baseline_no_semi'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} --epoch 40 --start_rl_epoch 0 --batch 2 --img_dir /home/tp030/CT/notes/chapter_11_will_ISP_be_able_to_do_it/IndividualScanpath/COCO_Search18/ourdata_cocoformat --feat_dir /home/tp030/CT/notes/chapter_11_will_ISP_be_able_to_do_it/IndividualScanpath/COCO_Search18/ourdata_cocoformat/swin_unetr_feature_phong 

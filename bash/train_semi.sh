# The name of this experiment.
DATASET_NAME='COCO_Search'
MODEL_NAME='baseline_semi'

# Save logs and models under snap/; make backup.
output=runs/${DATASET_NAME}_${MODEL_NAME}
mkdir -p $output/src
mkdir -p $output/bash
rsync -av  src/* $output/src/
cp $0 $output/bash/run.bash

CUDA_VISIBLE_DEVICES=1,2 python src/train.py \
  --log_root runs/${DATASET_NAME}_${MODEL_NAME} --epoch 40 --start_rl_epoch 40 --no_eval_epoch 40 --batch 8 --img_dir /home/tp030/CT/notes/chapter_4_semi_supervised_gazeformer3D/Gazeformer/egd_reflacx_coco --feat_dir /home/tp030/CT/notes/chapter_4_semi_supervised_gazeformer3D/Gazeformer/egd_reflacx_coco/swin_unetr_feature_phong 

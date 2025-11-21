#!/bin/bash
#SBATCH --job-name=CTSearcher
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%N_%j.out
#SBATCH --error=slurm_logs/%N_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --distribution=cyclic
# #SBATCH --partition=gpu
# #SBATCH --constraint=a100

mkdir -p slurm_logs

# Run training - Lightning auto-detects Slurm and configures DDP
srun --nodes=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 \
  python src/train_lightning.py \
    --log_root runs/CTScanGaze_CTSearcher \
    --epoch 40 \
    --batch 2 \
    --img_dir /path/to/CTScanGaze/data \
    --feat_dir /path/to/CTScanGaze/features/swin_unetr_feature


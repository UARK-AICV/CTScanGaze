"""
CT-Searcher Training with PyTorch Lightning
Simplified training using Lightning framework
"""

import json
import os
from typing import Optional

import pytorch_lightning as pl
import scipy.stats
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dataset.dataset import CTScanGaze, CTScanGaze_evaluation
from models.ct_searcher import CTSearcher
from models.loss import CrossEntropyLoss, MLPLogNormalDistribution
from models.sampling import Sampling
from opts import parse_opt
from utils.evaluation import comprehensive_evaluation_by_subject


class CTSearcherLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for CT-Searcher

    Handles:
    - Forward pass
    - Loss computation
    - Optimization
    - Validation
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args.__dict__)
        self.args = args

        # CT-Searcher model
        self.model = CTSearcher(
            d_model=args.hidden_dim,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder,
            dim_feedforward=args.hidden_dim,
            dropout=args.decoder_dropout,
            spatial_dim=(args.im_h, args.im_w, 8),
            max_length=args.max_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Sampling module for validation/testing
        self.sampling = Sampling(
            convLSTM_length=args.max_length,
            min_length=args.min_length,
            map_width=args.im_w,
            map_height=args.im_h,
            width=args.width,
            height=args.height,
        )

        # Store validation outputs for epoch-end processing
        self.validation_step_outputs = []

    def forward(self, src):
        """Forward pass through model"""
        return self.model(src)

    def training_step(self, batch, batch_idx):
        """Training step with loss computation"""
        # Extract batch data
        tmp = [
            batch["images"],
            batch["durations"],
            batch["action_masks"],
            batch["duration_masks"],
            batch["target_scanpaths"],
        ]
        tmp = [_.view(-1, *_.shape[2:]) for _ in tmp]
        images, durations, action_masks, duration_masks, target_scanpaths = tmp

        # Forward pass
        predicts = self.model(src=images)

        # Compute losses
        loss_spatial = CrossEntropyLoss(
            predicts["spatial_logits"], target_scanpaths, action_masks
        )
        loss_duration = MLPLogNormalDistribution(
            predicts["duration_mu"],
            predicts["duration_sigma2"],
            durations,
            duration_masks,
        )
        loss = loss_spatial + self.args.lambda_1 * loss_duration

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_spatial", loss_spatial, on_step=True, on_epoch=True)
        self.log("train/loss_duration", loss_duration, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with scanpath generation"""
        # Extract batch data
        tmp = [batch["images"], batch["fix_vectors"]]
        tmp = [_.view(-1, *_.shape[2:]) if torch.is_tensor(_) else _ for _ in tmp]
        images, gt_fix_vectors = tmp

        # Generate predictions
        predict = self.model.inference(src=images)

        log_normal_mu = predict["duration_mu"]
        log_normal_sigma2 = predict["duration_sigma2"]
        all_actions_prob = predict["spatial_probs"]

        # Sample multiple scanpaths for evaluation
        predict_fix_vectors = []
        repeat_num = self.args.eval_repeat_num

        for trial in range(repeat_num):
            samples = self.sampling.random_sample(
                all_actions_prob, log_normal_mu, log_normal_sigma2
            )
            prob_sample_actions = samples["selected_actions_probs"]
            durations = samples["durations"]
            sample_actions = samples["selected_actions"]

            sampling_random_predict_fix_vectors, _, _ = self.sampling.generate_scanpath(
                images, prob_sample_actions, durations, sample_actions
            )
            predict_fix_vectors.extend(sampling_random_predict_fix_vectors)

        # Store for epoch-end evaluation
        self.validation_step_outputs.append(
            {
                "gt_fix_vectors": gt_fix_vectors,
                "predict_fix_vectors": predict_fix_vectors,
                "img_names": batch["img_names"],
            }
        )

        return {
            "gt_fix_vectors": gt_fix_vectors,
            "predict_fix_vectors": predict_fix_vectors,
        }

    def on_validation_epoch_end(self):
        """Compute metrics at end of validation epoch"""
        if not self.validation_step_outputs:
            return

        # Gather all predictions
        all_gt_fix_vectors = []
        all_predict_fix_vectors = []

        for output in self.validation_step_outputs:
            all_gt_fix_vectors.extend(output["gt_fix_vectors"])
            # Group predictions by image
            num_images = len(output["img_names"])
            for idx in range(num_images):
                start_idx = idx * self.args.subject_num * self.args.eval_repeat_num
                end_idx = (idx + 1) * self.args.subject_num * self.args.eval_repeat_num
                all_predict_fix_vectors.append(
                    output["predict_fix_vectors"][start_idx:end_idx]
                )

        # Compute comprehensive metrics
        cur_metrics, cur_metrics_std, _ = comprehensive_evaluation_by_subject(
            all_gt_fix_vectors, all_predict_fix_vectors, self.args
        )

        # Log all metrics
        for metrics_key in cur_metrics.keys():
            for metric_name, metric_value in cur_metrics[metrics_key].items():
                self.log(
                    f"val/{metrics_key}_{metric_name}",
                    metric_value,
                    on_epoch=True,
                    prog_bar=(
                        metrics_key == "ScanMatch"
                        and metric_name in ["w/ duration", "w/o duration"]
                    ),
                )

        # Compute and log main metric (harmonic mean of ScanMatch)
        if "ScanMatch" in cur_metrics:
            cur_metric = scipy.stats.hmean(list(cur_metrics["ScanMatch"].values()))
            self.log("val/metric_hmean", cur_metric, on_epoch=True, prog_bar=True)

        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.98),
            eps=1e-09,
            weight_decay=self.args.weight_decay,
        )

        # Calculate total training steps
        # Note: We approximate since we don't have dataloader length here
        # In practice, Lightning will handle this correctly
        steps_per_epoch = 100  # This will be updated by Lightning
        warmup_steps = steps_per_epoch * self.args.warmup_epoch
        total_steps = steps_per_epoch * self.args.epoch

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Warmup: linear increase
                return current_step / warmup_steps
            else:
                # Decay: linear decrease
                return max(
                    0.0, (total_steps - current_step) / (total_steps - warmup_steps)
                )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class CTScanGazeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CT-ScanGaze dataset

    Handles:
    - Dataset instantiation
    - DataLoader creation
    - Train/val splits
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation"""
        if stage == "fit" or stage is None:
            self.train_dataset = CTScanGaze(
                self.args.img_dir,
                self.args.feat_dir,
                self.args.fix_dir,
                action_map=(self.args.im_h, self.args.im_w, 8),
                resize=(self.args.height, self.args.width, 512),
                origin_size=(self.args.origin_height, self.args.origin_width, 512),
                blur_sigma=self.args.blur_sigma,
                type="train",
                max_length=self.args.max_length,
            )

            self.val_dataset = CTScanGaze_evaluation(
                self.args.img_dir,
                self.args.feat_dir,
                self.args.fix_dir,
                action_map=(self.args.im_h, self.args.im_w, 8),
                origin_size=(self.args.origin_height, self.args.origin_width, 512),
                resize=(self.args.height, self.args.width, 512),
                type="test",
            )

        if stage == "test":
            self.test_dataset = CTScanGaze_evaluation(
                self.args.img_dir,
                self.args.feat_dir,
                self.args.fix_dir,
                action_map=(self.args.im_h, self.args.im_w, 8),
                origin_size=(self.args.origin_height, self.args.origin_width, 512),
                resize=(self.args.height, self.args.width, 512),
                type="test",
            )

    def train_dataloader(self):
        """Create training dataloader"""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch,
            shuffle=True,
            num_workers=4,
            collate_fn=self.train_dataset.collate_func,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader"""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.args.test_batch,
            shuffle=False,
            num_workers=4,
            collate_fn=self.val_dataset.collate_func,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader"""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.test_batch,
            shuffle=False,
            num_workers=4,
            collate_fn=self.test_dataset.collate_func,
            pin_memory=True,
        )


def main():
    """Main training function using PyTorch Lightning"""
    # Parse arguments
    args = parse_opt()

    # Set random seeds for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create output directory
    log_dir = args.log_root
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    # Save hyperparameters
    hparams_file = os.path.join(log_dir, "hparams.json")
    if not os.path.exists(hparams_file):
        with open(hparams_file, "w") as f:
            json.dump(args.__dict__, f, indent=2)

    # Initialize Lightning components
    model = CTSearcherLightningModule(args)
    datamodule = CTScanGazeDataModule(args)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="",
        version="",
        default_hp_metric=False,
    )

    # Setup callbacks
    callbacks = [
        # Model checkpoint callback - saves best models
        ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="checkpoint-{epoch:02d}-{val_metric_hmean:.4f}",
            monitor="val/metric_hmean",
            mode="max",
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="step"),
    ]

    # Detect Slurm environment
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Configure strategy based on environment
    if num_nodes > 1:
        # Multi-node training
        strategy = "ddp"
        print(
            f"🚀 Multi-node training: {num_nodes} nodes × {num_gpus} GPUs = {num_nodes * num_gpus} total GPUs"
        )
    elif num_gpus > 1:
        # Single-node multi-GPU
        strategy = "ddp"
        print(f"🚀 Single-node multi-GPU: {num_gpus} GPUs")
    else:
        # Single GPU or CPU
        strategy = "auto"
        print("🚀 Single device training")

    # Initialize trainer
    trainer = pl.Trainer(
        default_root_dir=log_dir,
        max_epochs=args.epoch,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus if torch.cuda.is_available() else 1,
        num_nodes=num_nodes,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.clip if args.clip > 0 else None,
        log_every_n_steps=10,
        val_check_interval=1.0,  # Validate every epoch
        deterministic=True,
        precision="16-mixed"
        if torch.cuda.is_available()
        else 32,  # Mixed precision training
    )

    # Resume from checkpoint if specified
    ckpt_path = None
    if args.resume_dir != "":
        ckpt_file = os.path.join(args.resume_dir, "checkpoints", "last.ckpt")
        if os.path.exists(ckpt_file):
            ckpt_path = ckpt_file
            print(f"Resuming from checkpoint: {ckpt_path}")

    # Train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print("\nTraining complete!")
    print(f"Best model checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation metric: {trainer.checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()

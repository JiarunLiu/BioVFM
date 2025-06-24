
import os
import warnings
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import PyTorchProfiler, AdvancedProfiler, SimpleProfiler

from MyLitModel import MyLightningModel
from MyEvaluator import MyEvaluator
from MyEvaluatorBootstrap import MyEvaluatorBootstrap

import medmnist
import pandas as pd
import numpy as np


def setup_output(args):
    """ setup default save dirs, it can be overwritten by the user """
    # when training from scratch
    if args.save_dir is None and not os.path.isfile(args.pretrain_model):
        args.save_dir = f"output/{args.model_flag}/{args.data_flag}/{args.img_size}"
    # when using pretrain model, save dir is under the pretrain model folder
    elif args.save_dir is None:
        base_dir, model_name = os.path.split(args.pretrain_model)
        ft_type = "finetune" if args.enable_finetune else "linear_prob"
        if args.img_size != 224:
            ft_type = f"{ft_type}_{args.img_size}"
        model_name = model_name.replace(".pth", "")
        args.save_dir = os.path.join(base_dir, f"{ft_type}/{model_name}/{args.data_flag}")

    os.makedirs(args.save_dir, exist_ok=True)

    return args


def setup_logger(args):
    if args.wandb_tags is not None:
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = [args.model_flag, args.data_flag]

    # when training from scratch
    if args.run_name is None and not os.path.isfile(args.pretrain_model):
        args.run_name = f"{args.model_flag}_{args.data_flag}_{args.img_size}"
    # when using pretrain model, run name is based on the pretrain model
    elif args.run_name is None:
        pretrain_name = os.path.dirname(args.pretrain_model).split("/")[-1]
        model_name = os.path.basename(args.pretrain_model).replace(".pth", "")
        args.run_name = f"{pretrain_name}_{model_name}_{args.data_flag}"
        args.wandb_tags.append(model_name)
        args.wandb_tags.append(pretrain_name)

    if args.wandb_group is None:
        args.wandb_group = args.data_flag

    if args.evaluate:
        args.wandb_mode = "disabled"

    logger = WandbLogger(
        name=args.run_name,
        save_dir=args.save_dir,
        project=args.wandb_project,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        group=args.wandb_group,
        mode=args.wandb_mode,
    )

    return args, logger


# seed
seed_everything(42, workers=True)
warnings.simplefilter(action='ignore', category=FutureWarning)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--model_ckpt", default=None, type=str)  # this is the finetuned model for evaluation
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--fast_dev_run", default=False, action="store_true")
parser.add_argument("--profiler", default=None, type=str, 
                    choices=["simple", "advanced"])
parser.add_argument("--compute_ci", action="store_true", default=False)
parser = MyLightningModel.add_argparse_args(parser)
args = parser.parse_args()

# setup batch size
assert args.global_batch_size % args.num_devices == 0, "Batch size should be divisible by number of devices."
args.local_batch_size = args.global_batch_size // args.num_devices
print(f"Batch size per device: {args.local_batch_size}")
print(f"Total batch size: {args.global_batch_size}")

# prepare output folder
args = setup_output(args)

# create logger
args, logger = setup_logger(args)

# callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.save_dir, "checkpoints"),
    monitor="val/loss",
    every_n_epochs=1,
    save_on_train_epoch_end=True,
)
lr_monitor = LearningRateMonitor(logging_interval='epoch')

callbacks = [
    checkpoint_callback,
    lr_monitor,
]

# trainer
trainer = Trainer(
    max_epochs=args.num_epochs,
    callbacks=callbacks,
    logger=logger,
    accelerator="gpu", 
    devices=args.num_devices, 
    log_every_n_steps=10,
    enable_progress_bar=True,
    profiler=args.profiler,
    fast_dev_run=args.fast_dev_run,
)


# model
model = MyLightningModel(**vars(args))

if not args.evaluate:
    print("Start training...")
    trainer.fit(model)
    print("Training finished.")

if args.num_devices > 1:
    print("Please use single GPU for testing. Multiple GPUs are not supported for testing.")
    print("Exit.")
    exit(0)

# we didnot use the test function because we needs to merge all predictions together
print("Start testing...")

# load the best model
if args.model_ckpt is not None and args.evaluate:
    # if args.model_ckpt is a dir not a file, then the model_ckpt is the only file in the folder with ".ckpt"
    if os.path.isdir(args.model_ckpt):
        file_in_dir = os.listdir(args.model_ckpt)
        ckpt_files = [f for f in file_in_dir if f.endswith(".ckpt")]
        assert len(ckpt_files) == 1, f"More than one .ckpt file in {args.model_ckpt}"
        args.model_ckpt = os.path.join(args.model_ckpt, ckpt_files[0])
    print(f"Use {args.model_ckpt} as model checkpoint.")
    ckpt_path = args.model_ckpt
else:
    ckpt_path = "best"

# make testing predict
pred_outputs = trainer.predict(
    model=model, 
    dataloaders=model.test_dataloader(),
    ckpt_path=ckpt_path,
)
predictions = [p[0] for p in pred_outputs]
predictions = torch.cat(predictions, dim=0)
pred_targets = [p[1] for p in pred_outputs]
pred_targets = torch.cat(pred_targets, dim=0)

# make validation predict, we use val set to compute the optimal threshold
val_outputs = trainer.predict(
    model=model,
    dataloaders=model.val_dataloader(),
    ckpt_path=ckpt_path,
)
val_predictions = [p[0] for p in val_outputs]
val_predictions = torch.cat(val_predictions, dim=0)
val_pred_targets = [p[1] for p in val_outputs]
val_pred_targets = torch.cat(val_pred_targets, dim=0)

# create evaluator
task = medmnist.INFO[args.data_flag]["task"]
if args.compute_ci:
    evaluator = MyEvaluatorBootstrap(args.data_flag, "test")
else:
    evaluator = medmnist.Evaluator(args.data_flag, "test")

# process predictions
assert np.allclose(evaluator.labels.astype("int"), pred_targets.detach().cpu().numpy().astype("int"))
if task == "multi-label, binary-class":
    predictions = torch.sigmoid(predictions).detach().cpu().numpy()
    val_predictions = torch.sigmoid(val_predictions).detach().cpu().numpy()
else:
    predictions = torch.softmax(predictions, dim=1).detach().cpu().numpy()
    val_predictions = torch.softmax(val_predictions, dim=1).detach().cpu().numpy()

# save predictions and labels to npy file
np.save(os.path.join(args.save_dir, "predictions.npy"), predictions)
np.save(os.path.join(args.save_dir, "labels.npy"), evaluator.labels)

np.save(os.path.join(args.save_dir, "val_predictions.npy"), val_predictions)
np.save(os.path.join(args.save_dir, "val_labels.npy"), val_pred_targets)

if args.compute_ci:
    print("Start computing CI...")
    boot_stats, cis, val_metrics, thresholds = evaluator.evaluate(predictions, val_predictions=val_predictions, val_targets=val_pred_targets)
    boot_stats.to_csv(os.path.join(args.save_dir, "boot_stats.csv"))
    cis.to_csv(os.path.join(args.save_dir, "cis.csv"))
    thresholds.to_csv(os.path.join(args.save_dir, "thresholds.csv"))
    if val_metrics is not None:
        val_metrics.to_csv(os.path.join(args.save_dir, "val_metrics.csv"))
else:
    auc, acc = evaluator.evaluate(predictions, None, None)
    print(f"AUC: {auc}, ACC: {acc}")
    metrics = {
        f"{args.data_flag}_auc": auc,
        f"{args.data_flag}_acc": acc,
    }
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(args.save_dir, "metrics.csv"))


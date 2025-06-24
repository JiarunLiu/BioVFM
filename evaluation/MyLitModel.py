
import os
import argparse
from os.path import join
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import lr_scheduler, SGD, Adam, AdamW
from lightning import LightningModule
import torchvision.transforms as transforms
import torchmetrics

from timm.models.layers import trunc_normal_
from util.pos_embed import interpolate_pos_embed

from models.builder import build_model, AVAILABLE_MODELS

import medmnist
from medmnist import INFO, Evaluator
import models.models_vit as models_vit




class MyLightningModel(LightningModule):

    def __init__(
            self,
            lr: float = 1e-3,
            gamma: float = 0.1,
            weight_decay: float = 1e-5,
            local_batch_size: int = 128,
            global_batch_size: int = 128,
            workers: int = 4,
            data_flag: str = "pathmnist",
            img_size: int = 224,
            data_aug: str = "v0",
            num_epochs: int = 100,
            model_flag: str = "dino_vit_base",
            global_pool: bool = False,
            pretrain_model: str = "",
            enable_finetune: bool = False,
            pretrain_data_flag: str = 'pmcoa',  # determin the mean and std for image normalization
            optim_type: str = "adamw",
            **kwargs,
        ):
        super().__init__()
        
        self.lr = lr
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.local_batch_size = local_batch_size
        self.global_batch_size = global_batch_size
        assert global_batch_size / kwargs['num_devices'] == local_batch_size, "Global batch size should be divisible by number of devices."
        self.workers = workers
        self.num_epochs = num_epochs
        self.model_flag = model_flag
        self.optim_type = optim_type
        
        # collect dataset information
        self.info = INFO[data_flag]
        self.task = self.info['task']
        self.n_channels = 3
        self.n_classes = len(self.info['label'])
        self.load_train_data = kwargs['load_train_data']

        # create dataset
        self.data_flag = data_flag
        self.data_aug = data_aug
        self.img_size = img_size
        self.create_dataset()
        self.create_evaluater(data_flag)

        # create model
        self.global_pool = global_pool
        self.pretrain_model = pretrain_model
        self.enable_finetune = enable_finetune
        self.create_model()

        # create metric
        self.create_metric(task=self.task)

        self.save_hyperparameters()

    def create_model(self):
        # build model & load pretrain checkpoint & setup linear probing or fine-tuning
        self.model = build_model(
            model_flag=self.model_flag,
            n_class=self.n_classes,
            pretrain_model=self.pretrain_model,
            enable_finetune=self.enable_finetune,
            global_pool=self.global_pool,
            img_size=self.img_size,
            data_flag=self.data_flag,
        )

    def create_metric(self, task):
        if task == "multi-label, binary-class":
            # self.metric = torchmetrics.AUROC()
            self.train_acc = torchmetrics.Accuracy(task="multilabel", num_labels=self.n_classes)
            self.test_acc = torchmetrics.Accuracy(task="multilabel", num_labels=self.n_classes)
            self.val_acc = torchmetrics.Accuracy(task="multilabel", num_labels=self.n_classes)
            self.train_auc = torchmetrics.AUROC(task="multilabel", num_labels=self.n_classes)
            self.test_auc = torchmetrics.AUROC(task="multilabel", num_labels=self.n_classes)
            self.val_auc = torchmetrics.AUROC(task="multilabel", num_labels=self.n_classes)
        elif task == "binary-class":
            # assert self.num_classes == 2
            self.train_acc = torchmetrics.Accuracy(task="binary")
            self.test_acc = torchmetrics.Accuracy(task="binary")
            self.val_acc = torchmetrics.Accuracy(task="binary")
            self.train_auc = torchmetrics.AUROC(task="binary")
            self.test_auc = torchmetrics.AUROC(task="binary")
            self.val_auc = torchmetrics.AUROC(task="binary")
        elif task == "multi-class" or task == 'ordinal-regression':
            self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.n_classes)
            self.train_auc = torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes)
            self.test_auc = torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes)
            self.val_auc = torchmetrics.AUROC(task="multiclass", num_classes=self.n_classes)
        else:
            raise NotImplementedError

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser):
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--gamma", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0)
        parser.add_argument("--local_batch_size", type=int, default=-1)
        parser.add_argument("--global_batch_size", type=int, default=128)
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--num_devices", default=1, type=int)
        parser.add_argument("--optim_type", type=str, default="adamw", choices=["adam", "adamw", "sgd"])

        parser.add_argument("--num_epochs", type=int, default=100)
        parser.add_argument("--data_flag", type=str, default="pathmnist")
        parser.add_argument("--load_train_data", type=bool, default=True)
        parser.add_argument("--skip_train_data", action='store_false', dest='load_train_data', help="Skip loading train data")

        parser.add_argument("--img_size", type=int, default=224)
        parser.add_argument("--data_aug", type=str, default="v0", choices=["v0",])
        parser.add_argument("--model_flag", type=str, default="dino_vit_base", 
                            choices=list(AVAILABLE_MODELS.keys()))
        parser.add_argument('--global_pool', action='store_true')
        parser.set_defaults(global_pool=False)
        parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                            help='Use class token instead of global pool for classification')
        parser.add_argument("--pretrain_model", type=str, default="")
        parser.add_argument("--enable_finetune", action='store_true')
        parser.set_defaults(enable_finetune=False)
        parser.add_argument("--linear_prob", action='store_false', dest='enable_finetune')
        
        # wandb logger
        parser.add_argument("--run_name", type=str, default=None)
        parser.add_argument("--wandb_tags", type=str, default=None)
        parser.add_argument("--wandb_notes", type=str, default=None)
        parser.add_argument("--wandb_group", type=str, default=None)
        parser.add_argument("--wandb_project", type=str, default="eval_pmcoa")
        parser.add_argument("--wandb_mode", default="online", type=str, choices=["online", "offline", 'disabled'])

        parser.add_argument("--save_dir", type=str, default=None)

        return parser
    
    def data_aug_v0(self):
        img_size = self.img_size
        train_transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        return train_transform, test_transform
        
    def create_dataset(self):
        DataClass = getattr(medmnist, self.info['python_class'])
        img_size = self.img_size
        if self.data_aug == "v0":
            train_transform, test_transform = self.data_aug_v0()
        elif self.data_aug == "v0_1":
            train_transform, test_transform = self.data_aug_v0_1()
        elif self.data_aug == "v1":
            train_transform, test_transform = self.data_aug_v1()
        elif self.data_aug == "v2":
            train_transform, test_transform = self.data_aug_v2()
        elif self.data_aug == "simple":
            train_transform, test_transform = self.data_aug_simple()
        else:
            raise ValueError("Invalid data_aug_ver")
        if self.load_train_data:
            self.train_dataset = DataClass(split='train', transform=train_transform, download=False, size=img_size, as_rgb=True, mmap_mode='r')
        else:
            self.train_dataset = None
        self.eval_dataset = DataClass(split='val', transform=test_transform, download=False, size=img_size, as_rgb=True, mmap_mode='r')
        self.test_dataset = DataClass(split='test', transform=test_transform, download=False, size=img_size, as_rgb=True, mmap_mode='r')

    def create_evaluater(self, data_flag):
        if self.load_train_data:
            self.train_evaluator = medmnist.Evaluator(data_flag, 'train')
        else:
            self.train_evaluator = None
        self.val_evaluator = medmnist.Evaluator(data_flag, 'val')
        self.test_evaluator = medmnist.Evaluator(data_flag, 'test')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, targets = batch
        
        outputs = self.model(inputs)

        # gather outputs and targets from all devices
        outputs = self.all_gather(outputs)
        targets = self.all_gather(targets)
        if len(outputs.shape) > 2:
            outputs = outputs.reshape(-1, *outputs.shape[2:])
            targets = targets.reshape(-1, *targets.shape[2:])
        
        return outputs, targets, batch_idx, dataloader_idx

    def shared_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == 'multi-label, binary-class':
            targets = targets.type_as(outputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            targets = targets.long()
        else:
            targets = torch.squeeze(targets, 1).type_as(outputs).long()
            loss = F.cross_entropy(outputs, targets)

        if self.task == 'multi-label, binary-class':
            outputs = torch.sigmoid(outputs)
        else:
            outputs = F.softmax(outputs, dim=1)

        if self.task == "binary-class":
            if outputs.ndim == 2:
                outputs = outputs[:, -1]
            else:
                assert outputs.ndim == 1

        return loss, outputs, targets

    def training_step(self, batch, batch_idx):
        batch_size = batch[1].shape[0]
        loss_train, outputs, targets = self.shared_step(batch)
        self.train_acc(outputs, targets)
        self.train_auc(outputs, targets)
        self.log('train/loss', loss_train, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss_train

    def validation_step(self, batch, batch_idx):
        prefix = "val"
        batch_size = batch[1].shape[0]
        loss_eval, outputs, targets = self.shared_step(batch)

        self.val_acc(outputs, targets)
        self.val_auc(outputs, targets)

        self.log(f'{prefix}/loss', loss_eval, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/acc', self.val_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/auc', self.val_auc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss_eval
    
    def test_step(self, batch, batch_idx):
        prefix = "test"
        batch_size = batch[1].shape
        loss_eval, outputs, targets = self.shared_step(batch)

        self.test_acc(outputs, targets)
        self.test_auc(outputs, targets)

        self.log(f'{prefix}/loss', loss_eval, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/acc', self.test_acc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log(f'{prefix}/auc', self.test_auc, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss_eval

    def configure_optimizers(self):
        if self.optim_type.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_type.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optim_type.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        assert self.train_dataset is not None
        print("train_dataloader workers: ", self.workers)
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=self.workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.local_batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.local_batch_size,
            shuffle=False,
            num_workers=self.workers,
        )